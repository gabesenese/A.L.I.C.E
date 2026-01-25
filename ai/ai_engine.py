import numpy as np
import sys
import io
import logging
import os
import pickle
from typing import Optional, Tuple
from nlp_processor import NLPProcessor
from ai_model import AIModel, ModelTrainer
from dataset import DatasetLoader
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

# Encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")

# Set up logging
logging.basicConfig(
    encoding="utf-8",
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AIEngineConfig:
    """Configuration class for AI Engine parameters"""
    def __init__(
        self,
        svd_components: int = 300,
        test_split: float = 0.2,
        model_path: str = "models/alice_model.h5",
        svd_path: str = "models/svd_model.pkl",
        answers_path: str = "models/unique_answers.pkl"
    ):
        self.svd_components = svd_components
        self.test_split = test_split
        self.model_path = model_path
        self.svd_path = svd_path
        self.answers_path = answers_path

class AIEngine:
    def __init__(self, config: Optional[AIEngineConfig] = None):
        self.config = config or AIEngineConfig()
        self.nlp_processor = NLPProcessor()
        self.dataset_loader = None
        self.unique_answers = None
        self.svd = None
        self.ai_model = None
        
        # Try to load existing model, otherwise train new one
        if not self._load_models():
            logger.info("No existing models found. Training new model...")
            self._initialize_and_train()

    def _initialize_and_train(self):
        """Initialize dataset and train the model"""
        try:
            self.dataset_loader = DatasetLoader()
            
            # Prepare data
            X_reduced, y_numerical = self._prepare_data()
            
            # Initialize model
            input_dim = X_reduced.shape[1]
            output_dim = len(self.unique_answers)
            self.ai_model = AIModel(input_dim=input_dim, output_dim=output_dim)
            
            # Train and evaluate
            self._train_model(X_reduced, y_numerical)
            
            # Save models
            self._save_models()
            
        except Exception as e:
            logger.error(f"Error during initialization and training: {e}")
            raise

    def _prepare_data(self) -> Tuple[np.ndarray, list]:
        """Prepare and process training data"""
        try:
            # Convert answers to unique numerical indices
            self.unique_answers = list(set(self.dataset_loader.answers))
            y_numerical = [
                self.unique_answers.index(answer) 
                for answer in self.dataset_loader.answers
            ]

            # Vectorize questions
            X_vectorized = self.nlp_processor.vectorize(self.dataset_loader.questions)
            
            # Reduce dimensionality
            self.svd = TruncatedSVD(n_components=self.config.svd_components)
            X_reduced = self.svd.fit_transform(X_vectorized)
            
            logger.info(f"Data prepared - Shape: {X_reduced.shape}, Classes: {len(self.unique_answers)}")
            return X_reduced, y_numerical
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise

    def _train_model(self, X: np.ndarray, y: list):
        """Train the AI model with train/test split"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_split, random_state=42
            )
            
            logger.info(f'Train data shape: {X_train.shape}, Test data shape: {X_test.shape}')
            
            # Train model
            self.ai_model.train(X_train, y_train)
            
            # Evaluate on test set
            test_accuracy = self.ai_model.evaluate(X_test, y_test)
            logger.info(f'Test accuracy: {test_accuracy:.4f}')
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def _save_models(self):
        """Save trained models to disk"""
        try:
            os.makedirs("models", exist_ok=True)
            
            # Save AI model
            self.ai_model.save(self.config.model_path)
            
            # Save SVD model
            with open(self.config.svd_path, 'wb') as f:
                pickle.dump(self.svd, f)
            
            # Save unique answers
            with open(self.config.answers_path, 'wb') as f:
                pickle.dump(self.unique_answers, f)
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise

    def _load_models(self) -> bool:
        """Load existing models from disk"""
        try:
            if not all([
                os.path.exists(self.config.model_path),
                os.path.exists(self.config.svd_path),
                os.path.exists(self.config.answers_path)
            ]):
                return False
            
            # Load unique answers
            with open(self.config.answers_path, 'rb') as f:
                self.unique_answers = pickle.load(f)
            
            # Load SVD model
            with open(self.config.svd_path, 'rb') as f:
                self.svd = pickle.load(f)
            
            # Load AI model
            input_dim = self.config.svd_components
            output_dim = len(self.unique_answers)
            self.ai_model = AIModel(input_dim=input_dim, output_dim=output_dim)
            self.ai_model.load(self.config.model_path)
            
            logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logger.warning(f"Error loading models: {e}")
            return False

    def process_input(self, user_input: str) -> str:
        """Process user input and generate response"""
        try:
            # Validate input
            if not user_input or not user_input.strip():
                return "I didn't catch that. Could you say that again?"
            
            # Transform and vectorize input
            transformed_input = self.nlp_processor.transform(user_input)
            logger.debug(f"Transformed input shape: {transformed_input.shape}")
            
            # Reduce dimensionality
            reduced_input = self.svd.transform(transformed_input)
            logger.debug(f"Reduced input shape: {reduced_input.shape}")

            # Get prediction
            prediction = self.ai_model.predict(reduced_input)
            logger.debug(f"Prediction probabilities: {prediction}")

            # Get response
            response_index = np.argmax(prediction)
            confidence = prediction[0][response_index]
            logger.info(f"Predicted response index: {response_index} (confidence: {confidence:.4f})")
            
            # Return response with confidence threshold
            if confidence < 0.3:
                return "I'm not sure I understand. Could you rephrase that?"
            
            return self.unique_answers[response_index]
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return "I encountered an error processing your request. Please try again."

    def retrain(self):
        """Retrain the model with current dataset"""
        logger.info("Retraining model...")
        self._initialize_and_train()

# Example usage
if __name__ == "__main__":
    try:
        # Initialize with custom config if needed
        config = AIEngineConfig(svd_components=300, test_split=0.2)
        ai_engine = AIEngine(config)
        
        print("A.L.I.C.E: Hi there, how can I help you?")
        print("(Type 'exit' or 'quit' to end the conversation)")
        
        while True:
            try:
                user_input = input("\nUser: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("A.L.I.C.E.: Goodbye! Have a great day!")
                    break

                response = ai_engine.process_input(user_input)
                print(f"A.L.I.C.E.: {response}")
                
            except KeyboardInterrupt:
                print("\nA.L.I.C.E.: Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in conversation loop: {e}")
                print("A.L.I.C.E.: Sorry, I encountered an error. Let's try again.")
                
    except Exception as e:
        logger.error(f"Failed to initialize AI Engine: {e}")
        print("Failed to start A.L.I.C.E. Please check the logs.")
