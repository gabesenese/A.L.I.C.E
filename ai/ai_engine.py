import numpy as np
import sys
import io
import logging
from nlp_processor import NLPProcessor
from ai_model import AIModel, ModelTrainer
from dataset import DatasetLoader
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD


# Encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf=8")
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")


# Set up logging
logging.basicConfig(encoding="utf-8", level=logging.INFO)

class AIEngine:
    def __init__(self):
        self.nlp_processor = NLPProcessor()
        self.dataset_loader = DatasetLoader()

        # Convert answers to unique numerical indices
        self.unique_answers = list(set(self.dataset_loader.answers))
        y_numerical = [self.unique_answers.index(answer) for answer in self.dataset_loader.answers]

        # Vectorize questions and reduce dimensionality
        X_vectorized = self.nlp_processor.vectorize(self.dataset_loader.questions)
        self.svd = TruncatedSVD(n_components=300)
        X_reduced = self.svd.fit_transform(X_vectorized)

        input_dim = X_reduced.shape[1]
        output_dim = len(self.unique_answers)
        self.ai_model = AIModel(input_dim=input_dim, output_dim=output_dim)

        # Train AI model
        self.train_model(X_reduced, y_numerical)

    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.ai_model.train(X_train, y_train)
        logging.info(f'Train data shape: {X_train.shape}, Test Data Shape: {X_test.shape}')
        #print(f'Train data shape: {X_train.shape}, Test Data Shape: {X_test.shape}')

    def process_input(self, user_input):
        transformed_input = self.nlp_processor.transform(user_input)
        logging.info(f"transformed input shape: {transformed_input}")
        
        reduced_input = self.svd.transform(transformed_input)
        logging.info(f"Reduced input shape: {reduced_input}")

        prediction = self.ai_model.predict(reduced_input)
        logging.info(f"Prediction: {prediction}")

        response_index = np.argmax(prediction)
        logging.info(f"Predicted response index: {response_index}")
        return self.unique_answers[response_index]

# Example usage
if __name__ == "__main__":
    ai_engine = AIEngine()
    print("A.L.I.C.E: Hi there, how can I help you?")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("A.L.I.C.E.: Bye for now!")
            break

        response = ai_engine.process_input(user_input)
        print(f"A.L.I.C.E.: {response}")
