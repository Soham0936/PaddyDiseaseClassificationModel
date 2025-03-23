🌾 Paddy Disease Classification Model

📌 Overview

The Paddy Disease Classification Model leverages transfer learning with DenseNet121 to accurately classify paddy leaf diseases. The model achieves 97% training accuracy and 95% testing accuracy, making it a robust and reliable solution for early disease detection in paddy crops. 🌱📊

📂 Dataset

🖼 Training Data: 10,407 labeled images across 9 disease classes and healthy leaves.

🔍 Test Data: 3,469 unlabeled images.

📌 Source: Kaggle - Paddy Disease Classification

⚙️ Methodology

🛠 Data Preparation

✅ Load and preprocess the dataset.
✅ Split into train, validation, and test sets.
✅ Normalize pixel values.
✅ Apply data augmentation (rotation, flipping, zoom, etc.) to enhance generalization.

🏗 Model Architecture

🔹 Use DenseNet121 as the base model (pre-trained on ImageNet).
🔹 Add custom layers:

🏗 Batch Normalization

🛑 Dropout (to prevent overfitting)

🔢 Fully Connected Dense layers
🔹 Compile with Adam optimizer and sparse categorical cross-entropy loss.

🚀 Training Process

⚡ Train using transfer learning.
⚡ Apply early stopping to prevent overfitting.
⚡ Achieves 95% validation accuracy.
⚡ Track training progress through loss and accuracy plots.

📊 Evaluation

📈 Compute classification metrics: accuracy, precision, recall, F1-score.
📉 Visualize training/validation accuracy and loss curves.

🔍 Prediction & Clustering

🤖 Classify unseen images using the trained model.
📌 Utilize K-Means clustering for further pattern analysis.

🛠 Tools & Technologies

🧠 TensorFlow/Keras: Model building and training.

📊 NumPy/Pandas: Data preprocessing and manipulation.

📈 Matplotlib/Seaborn: Data visualization.

🔍 Scikit-learn: K-Means clustering.

📁 Repository Structure

Paddy-Doctor/
├── 📜 README.md                # Project documentation
├── 📓 Paddy_Doctor.ipynb       # Jupyter Notebook with model code
├── 📄 requirements.txt         # Dependencies
├── 🖼 train_images/            # Training dataset
└── 🖼 test_images/             # Test dataset

🔧 Installation & Usage

📌 Prerequisites

Ensure you have Python 3.8+ and the required dependencies installed:

pip install -r requirements.txt

▶️ Running the Model

Run the Jupyter Notebook to train and evaluate the model:

jupyter notebook Paddy_Doctor.ipynb

📊 Results

✔ Training Accuracy: 97%
✔ Testing Accuracy: 95%
✔ Model successfully identifies diseases with high accuracy, helping farmers detect issues early and take preventive measures. 🌱💡

🚀 Future Enhancements

🔹 Expand dataset for improved generalization.
🔹 Integrate with mobile/web applications for real-time disease detection.
🔹 Implement explainable AI techniques for better model interpretability.

