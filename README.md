# Paddy Disease Classification Model
Paddy Disease Classification
Overview
This project uses transfer learning with DenseNet121 to classify paddy leaf diseases. The model achieves 97% training accuracy and 95% testing accuracy, making it a robust solution for identifying diseases in paddy crops.

Dataset
Training Data: 10,407 labeled images (9 diseases + healthy leaves).

Test Data: 3,469 unlabeled images.

Source: Kaggle Paddy Disease Classification.

Methodology
Data Preparation:

Load, preprocess, and split data into train/validation/test sets.

Normalize pixel values and apply data augmentation.

Model Building:

Use DenseNet121 as the base model.

Add custom layers (Batch Normalization, Dropout, Dense).

Compile with Adam optimizer and sparse categorical cross-entropy loss.

Training:

Train using transfer learning with early stopping.

Evaluation:

Achieves 95% validation accuracy.

Visualize training/validation accuracy and loss.

Prediction:

Predict diseases on unseen data using K-Means clustering.

Tools
TensorFlow/Keras: Model building and training.

NumPy/Pandas: Data manipulation.

Matplotlib/Seaborn: Visualization.

Scikit-learn: K-Means clustering.

Repository Structure
Copy
Paddy-Doctor/
├── README.md
├── Paddy_Doctor.ipynb
├── requirements.txt
├── train_images/  # Training dataset
└── test_images/   # Test dataset
