# GTSRB Traffic Sign Classifier

Traffic sign classification on the German Traffic Sign Recognition Benchmark (GTSRB) using a convolutional neural network built with PyTorch.

This project walks through the full workflow of building an image classifier for autonomous driving perception: exploring the data, designing and training a CNN, analyzing its mistakes, improving it step by step, and preparing it for deployment on edge hardware.

## Dataset

The project uses the [German Traffic Sign Dataset](https://www.kaggle.com/datasets/harbhajansingh21/german-traffic-sign-dataset) from Kaggle, a preprocessed version of GTSRB:

- 43 traffic sign classes
- 32x32 RGB images
- Separate training, validation and test splits

The dataset is downloaded automatically with `kagglehub` when the notebook runs.

## Approach

1. **Data exploration:** Inspecting image quality, lighting conditions and class distribution to understand what makes the task difficult.
2. **Data preparation:** Normalization, conversion to PyTorch tensors and batching with shuffling.
3. **Baseline model:** A small CNN trained from scratch to establish a reference point.
4. **Error analysis:** Per-class accuracy and confusion analysis to find where and why the model fails.
5. **Model improvements:** A deeper architecture with batch normalization and dropout to reduce overfitting.
6. **Data augmentation:** Random rotation, translation, scaling and brightness/contrast changes to make the model robust to real-world camera conditions.
7. **Robustness evaluation:** Comparing models on both the standard test set and a harder, augmented test set.
8. **Deployment preparation:** Exporting the model for efficient inference on edge devices.

## Repository Structure

```
.
├── traffic_sign_classifier.ipynb   # Full workflow: data, models, training, evaluation
├── .gitignore
└── README.md
```

Trained model weights (`.pt` files) are not included in the repository; they are produced by running the notebook.

## Getting Started

**Requirements:** Python 3.9+

```bash
git clone git@github.com:yigitScripts/gtsrb-traffic-sign-cnn.git
cd gtsrb-traffic-sign-cnn

python3 -m venv venv
source venv/bin/activate
pip install torch torchvision numpy pandas matplotlib kagglehub jupyter
```

Then open `traffic_sign_classifier.ipynb` and run the cells in order.

The notebook automatically selects the best available device: CUDA (NVIDIA GPU), MPS (Apple Silicon GPU) or CPU.

## Tech Stack

- **Python**
- **PyTorch** and **torchvision**: model, training and data augmentation
- **NumPy** and **pandas**: data handling
- **Matplotlib**: visualization
- **kagglehub**: dataset download
