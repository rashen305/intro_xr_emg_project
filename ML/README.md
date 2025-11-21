# ⚡️ Machine Learning Subdirectory
This folder contains all logic related to training and evaluating a neural network to statically recognize different gestures.
We additionally provide for real-time inferencing based on incoming streaming EMG sensor data via TCP socket.

---

## 📂 Project File Tree
The project structure assumes the core Python modules and notebooks are contained within an `ml/` subdirectory, and the raw data is stored in a `myo/samples/` folder.

```
intro_xr_emg_project/
├── ml/
│   ├── cnn_model.py                # CNN Architecture & Training Module (PyTorch)
│   ├── emg_preprocessing.py        # Data Processing Logic (Filtering, STFT, Dataset)
│   ├── train.ipynb                 # Primary Training Notebook
│   ├── evaluate.ipynb              # Evaluation/Verification Notebook
│   ├── README.md                   # Project Documentation (This file)
│   ├── **init**.py                 # Makes 'ml' a Python package (Recommended)
│   └── normalization_params.npy    # 💾 GENERATED: Data Mean and Standard Deviation
├── myo/
│   └── samples/
│       ├── raymond_arm_90_deg_200hz.csv       # 📊 Input Data File (Example)
│       └── raymond_arm_down_pinch_200hz.csv   # 📊 Input Data File (Example)
├── emg-to-pytorch.cpp              # 🖥️ Live Data Acquisition (C++ Myo SDK)
└── [Other Project Files...]
```

---

## 💡 File Descriptions

| File / Module | Function | Details |
| :--- | :--- | :--- |
| **`cnn_model.py`** | **Model & Training Module** | Defines the **`CNNmodel`** class, which contains the complete 2D-CNN architecture. Importantly, it includes integrated methods (`train_epoch`, `test_epoch`, and the high-level `train`) that encapsulate the optimization, loss, and full training loop logic. |
| **`emg_preprocessing.py`** | **Data Pipeline** | Contains all constant definitions, filtering methods (`preprocess`), and the PyTorch **`EMGDataset`** class. It ensures consistent EMG signal preprocessing (detrending, filtering, STFT) across all stages. |
| **`train.ipynb`** | **Training Script** | Loads data, calculates and saves **normalization parameters**, initializes the `CNNmodel`, and executes the full training process via `model.train()`. It saves the final model weights. |
| **`evaluate.ipynb`** | **Verification Script** | Loads a pre-trained model and normalization parameters to assess performance on new or test data and visualizes the results (Confusion Matrix). |
| **`normalization_params.npy`** | **Critical Metadata** | A NumPy binary file containing the **global mean ($\mu$) and standard deviation ($\sigma$)** calculated *only* from the training data. This must be used to scale all future input data. |
| **`*.pth`** | **Model Output** | The file containing the PyTorch **state dictionary** (weights and biases) of the trained `CNNmodel` after all epochs are complete. |

---

## 🚀 Execution Workflow
The project is designed for a modular, two-stage execution:

### 1. Training (Run `train_cnn.ipynb`)
This is the only stage where the model learns and where the normalization parameters are calculated.

1.  Set configuration (`EPOCHS`, `LR`, etc.) in the notebook.
2.  Data is preprocessed, and the global mean/std are computed.
3.  The **`normalization_params.npy`** file is generated.
4.  The model runs its full training cycle (`model.train(...)`).
5.  The final, fully trained model state is saved to **`train_single_subject_myo_model.pth`**.

### 2. Evaluation (Run `evaluate.ipynb`)
This stage verifies the model's generalization ability.

1.  The script loads **`normalization_params.npy`** and **`train_single_subject_myo_model.pth`**.
2.  Evaluation data is preprocessed and scaled using the loaded mean/std.
3.  The model runs its `test_epoch` to generate final accuracy and the Confusion Matrix.