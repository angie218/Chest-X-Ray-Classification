# Chest X-Ray Classification

A deep learning project for multi-class chest X-ray classification using three different neural network approaches: a custom Convolutional Neural Network (CNN), EfficientNetB0 transfer learning, and a patch-based Bidirectional GRU.

The project explores different strategies for distinguishing between **Normal**, **Bacterial Pneumonia**, and **Viral Pneumonia** chest X-ray images.

> **Disclaimer:** This project is intended for educational and research purposes only. It is not a medical diagnostic system and should not be used for clinical decision-making.

## Project Overview

The goal of this project is to compare multiple deep learning architectures for chest X-ray image classification.

Three modeling approaches were implemented:

### 1. Custom CNN

A Convolutional Neural Network trained from scratch to learn visual features directly from the chest X-ray images.

### 2. EfficientNetB0 Transfer Learning

A pretrained EfficientNetB0 model using ImageNet weights.

Training is performed in two stages:

- Frozen-backbone training
- Fine-tuning with a lower learning rate

This approach explores whether pretrained visual representations can improve performance on the chest X-ray classification task.

### 3. Patch-Based Bidirectional GRU

Chest X-ray images are divided into patches and represented as sequences.

A Bidirectional GRU processes these patch sequences in both directions to model relationships between different regions of the image.

## Dataset

The project uses the **Chest X-Ray Images (Pneumonia)** dataset available on Kaggle.

The dataset is downloaded programmatically using `kagglehub`:

```python
kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
```

Images are classified into three categories:

- Normal
- Bacterial Pneumonia
- Viral Pneumonia

The original test set is kept separate for final evaluation.

The dataset itself is not stored in this repository.

## Evaluation

The models are evaluated using several metrics and visualizations:

- Test accuracy
- Test loss
- Training and validation accuracy
- Training and validation loss
- Confusion matrices
- Precision
- Recall
- F1-score

Class weighting is also used during training to help account for differences in class distribution.

## Repository Structure

```text
Chest-X-Ray-Classification/
│
├── README.md
├── requirements.txt
├── .gitignore
│
└── notebooks/
    ├── Chest_XRay_Classification.ipynb
    └── Test_Environment.ipynb
```

### `Chest_XRay_Classification.ipynb`

Main experimentation notebook containing:

- Dataset acquisition and preprocessing
- Data organization and labeling
- Training and validation split
- Class balancing
- Custom CNN development
- EfficientNetB0 transfer learning
- Fine-tuning
- Patch-based Bidirectional GRU
- Model evaluation
- Model comparison

### `Test_Environment.ipynb`

Interactive inference environment for testing trained models on individual chest X-ray images.

The notebook allows an image to be uploaded and compares predictions from the trained architectures.

## Technologies

- Python
- TensorFlow / Keras
- EfficientNetB0
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn
- KaggleHub
- Google Colab

## Running the Project

Clone the repository:

```bash
git clone https://github.com/angie218/Chest-X-Ray-Classification.git
cd Chest-X-Ray-Classification
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Then open:

```text
notebooks/Chest_XRay_Classification.ipynb
```

The dataset is downloaded automatically through KaggleHub when the notebook is executed.

The main notebook creates local output directories for trained models, training histories, and comparison results.

## Test Environment

To test trained models on an individual chest X-ray image, open:

```text
notebooks/Test_Environment.ipynb
```

The test environment loads the trained models and runs the uploaded image through the available architectures, displaying predicted classes and confidence scores.

## Limitations and Future Work

Potential extensions include:

- Evaluation on additional external chest X-ray datasets
- Additional data augmentation and regularization
- Grad-CAM or other explainability techniques
- Systematic hyperparameter optimization
- Additional pretrained architectures
- Ensemble approaches
- More detailed analysis of class-specific errors

## Medical Use Disclaimer

The models in this repository were developed as part of a machine learning project and are intended solely for educational and research purposes.

They have **not** been clinically validated and must not be used to diagnose pneumonia or make medical decisions.
