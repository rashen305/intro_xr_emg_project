import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pickle # Required for serialization/deserialization

# The sklearn SVM model
class SVMModel:
    """
    Wrapper for a Scikit-learn Support Vector Classifier (SVC).
    This model expects the input to be a flattened feature vector (batch_size, 8).
    """
    def __init__(self, num_classes=2, kernel='rbf', device="cpu"):
        # We define a pipeline for robust classification that includes scaling.
        self.model = make_pipeline(
            StandardScaler(), # Feature scaling is crucial for SVM performance
            SVC(
                C=1.0,           # Regularization parameter
                kernel=kernel,   # 'rbf' (Radial Basis Function) is a common choice
                gamma='scale',   # Kernel coefficient
                probability=True # Allows prediction of probabilities (if needed)
            )
        )
        print(f"SVM Model Initialized (Kernel: {kernel})")

    # --- Training/Test Methods (Using sklearn's API) ---
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Trains the SVM model on NumPy arrays."""
        print("Starting SVM training...")
        self.model.fit(X_train, y_train)
        print("SVM training complete.")

    def test_epoch(self, X_test: np.ndarray, y_test: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluates the model on test data.
        """
        preds_all = self.model.predict(X_test)
        test_acc = self.model.score(X_test, y_test)
        
        return test_acc, preds_all, y_test
    
    # --- Model Persistence Methods ---
    def dump(self, filename: str):
        """
        Saves the entire scikit-learn pipeline (model and scaler) to a file 
        using Python's pickle serialization.
        """
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"✅ Model saved successfully to: {filename}")
        except Exception as e:
            print(f"❌ Error saving model to {filename}: {e}")

    @classmethod
    def load(cls, filename: str):
        """
        Loads a serialized scikit-learn pipeline from a file and wraps it 
        in a new SVMModel instance.
        """
        try:
            with open(filename, 'rb') as f:
                loaded_pipeline = pickle.load(f)
                
            # Create a new instance of SVMModel
            instance = cls()
            
            # Replace the default model with the loaded pipeline
            instance.model = loaded_pipeline
            print(f"✅ Model loaded successfully from: {filename}")
            return instance
            
        except FileNotFoundError:
            print(f"❌ Error: Model file not found at {filename}")
            return None
        except Exception as e:
            print(f"❌ Error loading model from {filename}: {e}")
            return None