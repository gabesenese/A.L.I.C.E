import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

class AIModel:
    def __init__(self, input_dim, output_dim):
        self.model = Sequential([
            Dense(512, input_shape=(input_dim,), activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(output_dim, activation='softmax' if output_dim > 2 else 'sigmoid')
        ])
        loss = 'sparse_categorical_crossentropy' if output_dim > 2 else 'binary_crossentropy'
        self.model.compile(optimizer=Adam(learning_rate=0.0001), loss=loss, metrics=['accuracy'])

    def train(self, X, y):
        X = np.array(X)
        y = np.array(y) 
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        self.model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    def predict(self, X):
        return self.model.predict(X)
    
class ModelTrainer:
    def __init__(self, questions, answers, unique_answers, vectorizer) :
        self.questions = questions
        self.answers = answers
        self.unique_answers = unique_answers
        self.vectorizer = vectorizer
        self.svd = TruncatedSVD(n_components=300)

    def preprocess_data(self):
        X_vectorized = self.vectorizer.fit_transform(self.questions)
        X_reduced = self.svd.fit_transform(X_vectorized)

        y_numerical = [self.unique_answers.index(answer) for answer in self.answers]

        return X_reduced, y_numerical
    

    def train_model(self):
        X, y = self.preprocess_data()
        input_dim = X.shape[1]
        output_dim = len(self.unique_answers)


        ai_model = AIModel(input_dim=input_dim, output_dim=output_dim)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        ai_model.train(X_train, y_train)

        print(f"Train data shape: {X_train.shape}, Test Data Shape: {X_test.shape}")


        return ai_model, self.svd
    

if __name__ == "__main__":
    from datasets.dataset import DatasetLoader
    from nlp_processor import NLPProcessor

    dataset_loader = DatasetLoader()
    nlp_processor = NLPProcessor()

    unique_answers = list(len(dataset_loader.answers))
    model_trainer = ModelTrainer(
        questions=dataset_loader.questions,
        answers=dataset_loader.answers,
        unique_answers=unique_answers,
        vectorizer=nlp_processor.vectorizer
    )

    ai_model, svd = model_trainer.train_model()



        
