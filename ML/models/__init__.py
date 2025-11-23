# models/__init__.py

# Import the specific classes/functions from the files in this directory
from .cnn_model import CNNModel
# from rnn_model import RNNModel

# Optional: Define __all__ to control what '*' imports pull in
__all__ = [
    "CNNModel",
    #"RNNModel"
]