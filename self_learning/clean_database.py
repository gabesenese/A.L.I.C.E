import sqlite3
import re

# Connect to the database
conn = sqlite3.connect('knowledge_base.db')
c = conn.cursor()

# List of unwanted phrases and patterns to filter out
unwanted_phrases = [
    "You can help Wikipedia by expanding it",
    "This article is a stub",
    "Retrieved from",
    "[", "]", "References"
]

# Function to clean and filter sentences based on predefined criteria
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

# Function to filter unwanted phrases and patterns from the database
def filter_database():
    for phrase in unwanted_phrases:
        c.execute("DELETE FROM knowledge WHERE sentence LIKE ?", ('%' + phrase + '%',))
        conn.commit()
        rows_deleted = c.rowcount
        if rows_deleted > 0:
            print(f"Deleted {rows_deleted} rows containing the phrase: '{phrase}'")

# Function to delete duplicate sentences from the database
def delete_duplicates():
    c.execute('''
        DELETE FROM knowledge
        WHERE id NOT IN (
            SELECT MIN(id)
            FROM knowledge
            GROUP BY sentence
        )
    ''')
    conn.commit()
    rows_deleted = c.rowcount
    if rows_deleted > 0:
        print(f"Deleted {rows_deleted} duplicate rows")
    else:
        print("No duplicate rows found to delete")

# Main function to execute cleaning operations
def main():
    filter_database()
    delete_duplicates()

if __name__ == "__main__":
    main()







