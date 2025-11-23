import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# The sklearn SVM model
class SVMModel:
    """
    Wrapper for a Scikit-learn Support Vector Classifier (SVC).
    This model expects the input to be a flattened feature vector (batch_size, 8).
    """
    def __init__(self, num_classes=2, kernel='rbf', device="cpu"):
        # Note: SVM training typically requires NumPy arrays, not PyTorch tensors.
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
        Note: We use the sklearn API which expects the full dataset for evaluation.
        """
        preds_all = self.model.predict(X_test)
        test_acc = self.model.score(X_test, y_test)
        
        return test_acc, preds_all, y_test