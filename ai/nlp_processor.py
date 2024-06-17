from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data and makes sure it is up to date
#nltk.download('punkt')

class NLPProcessor():
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def process(self, text):
        # tokenize text input
        tokens = word_tokenize(text)
        return tokens
    
    def vectorize(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, text):
        return self.vectorizer.transform([text])
           
