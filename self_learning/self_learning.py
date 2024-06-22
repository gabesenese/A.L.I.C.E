import os
import spacy
import requests
import sqlite3
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import time
from tqdm import tqdm
import re
import logging
from clean_database import unwanted_phrases
import clean_database
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# Initialize Logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize NLP
nlp = spacy.load('en_core_web_sm')

# Initialize the model and vectorizer
vectorizer = CountVectorizer()
model = MultinomialNB()

# Load or initialize the model and vectorizer
try:
    with open('model.pkl', 'rb') as f:
        model, vectorizer = pickle.load(f)
except FileNotFoundError:
    # Initial fit with some non-trivial data to prevent empty vocabulary error
    initial_data = ["This is some initial data to fit the vectorizer"]
    vectorizer.fit(initial_data)
    model.fit(vectorizer.transform(initial_data), ["initial response"])

# Set up SQLite database
conn = sqlite3.connect('knowledge_base.db')
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS knowledge (
        id INTEGER PRIMARY KEY,
        sentence TEXT,
        response TEXT
    )
''')
conn.commit()
  
# Function to clean and format sentences
def clean_sentence(sentence):
    sentence = sentence.strip()
    # Remove sentences with unwanted phrases
    if any(phrase in sentence for phrase in unwanted_phrases):
        return None
    # Remove sentences that are too short
    if len(sentence) < 20:
        return None
    # Remove sentences with only numbers or dates
    if re.match(r'^\d+$', sentence) or re.search(r'\b\d{4}\b', sentence):
        return None
    return sentence

# Check if sentence exists 
def sentence_exists(sentence):
    c.execute('SELECT COUNT(*) FROM knowledge WHERE sentence = ?', (sentence,))
    count = c.fetchone()[0]
    return count > 0

# Function to learn new responses and store in the database
def learn_response(question, answer):
    global model, vectorizer
    try:
        X = vectorizer.transform([question])
        y = [answer]
        model.partial_fit(X, y)
        with open('model.pkl', 'wb') as f:
            pickle.dump((model, vectorizer), f)
        
        # Check if the sentence already exists in the database
        if sentence_exists(question):
            logging.info(f"Duplicate found: '{question}'. Skipping...")
            return

        # Store in SQLite database
        c.execute('INSERT INTO knowledge (sentence, response) VALUES (?, ?)', (question, answer))
        conn.commit()

    except Exception as e:
        logging.error(f"Error learning response: {e}")

# Function to scrape data from Wikipedia and summarize the content
def scrape_data():
    url = "https://en.wikipedia.org/wiki/Special:Random"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the title of the page
        title = soup.find('h1', {'id': 'firstHeading'})
        if title:
            title = title.get_text()
        else:
            title = ""

        # Extract the full content
        paragraphs = soup.find_all('p')
        content = ' '.join([para.get_text() for para in paragraphs if para.get_text().strip()])

        # Summarize using sumy
        LANGUAGE = "english"
        parser = PlaintextParser.from_string(content, Tokenizer(LANGUAGE))
        stemmer = Stemmer(LANGUAGE)
        summarizer = LsaSummarizer(stemmer)
        summary = summarizer(parser.document, sentences_count=3)  # Adjust sentences_count if needed

        summarized_text = ' '.join([str(sentence) for sentence in summary])

        return title, summarized_text

    except requests.exceptions.RequestException as e:
        logging.error(f"Error scraping data from Wikipedia: {e}")
        return "", ""

# Function to update knowledge base
def update_knowledge_base():
    global model, vectorizer
    title, summarized_content = scrape_data()
    if not summarized_content.strip():
        logging.warning("Summarized content is empty, skipping...")
        return
    doc = nlp(summarized_content)
    sentences = list(doc.sents)
    if not sentences:
        logging.warning("No valid sentences found, skipping...")
        return
    logging.info("Learning new information...")
    for sentence in tqdm(sentences, desc="Learning Progress", unit="sentence"):
        try:
            learn_response(sentence.text, "This is newly learned information.")
        except Exception as e:
            logging.error(f"Error processing sentence '{sentence.text}': {e}")

# Function to process and learn from content (removed as requested)

def process_and_learn(content):
    doc = nlp(content)
    for sentence in doc.sents:
        question = clean_sentence(sentence.text)
        if not question:
            continue

        # Performed Name Entity Recognition => NER
        for ent in sentence.ents:
            if ent.label_ == "PERSON":
                learn_response(sentence.text, "This sentence mentions a person.")

            elif ent.label_ == "LOCATION":
                learn_response(sentence.text, "This sentence mentions a location.")

        # Incremental learning example
        try:
            X = vectorizer.transformer([sentence.text])
            model.partial_fit(X, ["Custom Response"])

        except Exception as e:
            logging.error(f"Error processing this sentence '{sentence.text}:' {e}")


# Main function
def main():
    while True:
        try:
            # Update knowledge base periodically
            update_knowledge_base()

            # Database cleaner
            clean_database()
            
            # Process & Learn content
            content = scrape_data()
            if not content.strip():
                continue

            process_and_learn(content)    
                
            # Sleep to avoid overwhelming the system with constant scraping
            time.sleep(5) # milliseconds
            
        except KeyboardInterrupt:
            logging.info("Script interrupted by user.")
            break
        except Exception as e:
            logging.error(f"Unexpected error in main loop: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
