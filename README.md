# XR EMG Project
This project seeks to perform real-time hand gesture identification, including pinching and force estimation.
We use an 8-channel sEMG for collecting signal data, machine learning for classification/inferencing, and Unity to demonstrate controls.

TCP sockets are used to communicate between the three segments of the pipeline.

`C++ (sEMG) → Python (neural inferencing) → C# (Unity scene)`


## 📂 Repository Structure
```
intro_xr_emg_project/   
├── README.md           # you are here
├── Unity/              # XR Scene for demonstrating gestures
├── data_transmission/  # Currently a scratch folder
├── environment.yaml    # Quickly install conda environment with all required python dependencies
├── ml/                 # All logic for training/evaluating/inferencing a neural network on sEMG data
├── myo/                # sEMG data collection through the Myo SDK
└── socket_folder/      # Currently a scratch folder
```

---

## 🧭 Overview
| Directory/File | Purpose |
| :--- | :--- |
| **`ml/`** | Contains the core PyTorch model, preprocessing scripts, and training/evaluation notebooks for EMG classification. |
| **`myo/`** | Stores raw EMG sample data and potentially Myo-specific SDK integration files. |
| **`Unity/`** | Contains the Unity project files, likely handling the 3D or XR visualization components. |
| **`environment.yaml`** | Conda or pip environment definition file, listing all required Python dependencies. |