# Fashion-MNIST Lab: Supervised & Semi-Supervised Deep Learning

This repository contains robust, modern pipelines for both supervised and semi-supervised learning on the Fashion-MNIST dataset. It features state-of-the-art transfer learning, data augmentation, advanced evaluation, and experiment tracking.

---

## 📦 Contents

- `fashionmnist_supervised.py` — Supervised learning pipeline using transfer learning (MobileNetV2/EfficientNetV2B0).
- `fashionmnist.py` — Semi-supervised learning pipeline with pseudo-labeling and fine-tuning.
- Data loading, preprocessing, augmentation, and visualization utilities.
- Advanced metrics: classification report, confusion matrix, ROC-AUC, and misclassification visualization.

---

## 🧠 Supervised Learning

- Uses MobileNetV2 or EfficientNetV2B0 as a base model (ImageNet weights).
- Freezes all but the last 10 layers of the base model for fine-tuning.
- Adds a custom head with dense, batch normalization, and dropout layers.
- Data augmentation: random flip, rotation, and zoom.
- Training monitored with EarlyStopping, ReduceLROnPlateau, and TensorBoard.
- Full evaluation: accuracy/loss curves, classification report, confusion matrix, ROC-AUC, and misclassified image visualization.

---

## 🤖 Semi-Supervised Learning

- Loads Fashion-MNIST from local CSV or Keras datasets.
- Splits data into labeled, unlabeled, validation, and test sets.
- Trains a model on a small labeled set.
- Uses the trained model to generate pseudo-labels for high-confidence unlabeled samples.
- Combines labeled and pseudo-labeled data for a second training stage (fine-tuning).
- Includes all the same evaluation and visualization tools as the supervised pipeline.

---

## 📊 Features

- Efficient `tf.data` pipelines for memory-friendly training.
- Data augmentation for better generalization.
- Experiment tracking with TensorBoard.
- Advanced evaluation: precision, recall, F1, confusion matrix, ROC-AUC.
- Visualization of class balance, dataset splits, and misclassified images.
- Easily switch between MobileNetV2 and EfficientNetV2B0.

---

## 🚀 Getting Started

1. Clone the repo and install requirements (TensorFlow, scikit-learn, matplotlib, seaborn, pandas).
2. Download Fashion-MNIST (automatically handled by Keras, or place CSVs in `fashion-mnist/`).
3. Run `fashionmnist_supervised.py` for supervised learning.
4. Run `fashionmnist.py` for semi-supervised learning.

---

## 📝 Customization

- Change the base model or fine-tuning strategy in the model creation function.
- Adjust augmentation, batch size, or learning rate as needed.
- For semi-supervised, set the number of labeled samples and pseudo-label confidence threshold.

---

## 📈 Example Results

- Training/validation accuracy and loss curves.
- Classification report and confusion matrix.
- ROC-AUC curves for each class.
- Visualization of the first 15 misclassified test images.

---

## 🤝 Contributing

Pull requests and suggestions are welcome! If you use this repo for research or teaching, please cite or star the project.


