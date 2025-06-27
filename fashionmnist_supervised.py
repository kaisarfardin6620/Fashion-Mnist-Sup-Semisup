import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import EfficientNetV2B0, MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras import regularizers
import os
import matplotlib.pyplot as plt
from math import ceil
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import seaborn as sns
import datetime

base_path = 'fashion-mnist'
img_height, img_width = 224, 224
batch_size = 32  
epochs = 10
num_classes = 10

tb_callback = TensorBoard(log_dir=f"logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6),
    tb_callback
]

print('Loading Fashion-MNIST from Keras datasets...')
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

def preprocess(image, label):
    image = tf.expand_dims(image, -1)  
    image = tf.image.grayscale_to_rgb(image)  
    image = tf.image.resize(image, [img_height, img_width])
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, num_classes)
    return image, label

print('Building tf.data datasets...')
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

print('Number of training samples:', x_train.shape[0])
print('Number of validation samples:', x_val.shape[0])
print('Number of testing samples:', x_test.shape[0])
print('Class indices: 0-9 (Fashion-MNIST standard)')

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

augmentation = tf.keras.Sequential([
    RandomFlip('horizontal'),
    RandomRotation(0.1),
    RandomZoom(0.1)
], name="augmentation")

print(f"Train shape: {x_train.shape}, Val shape: {x_val.shape}, Test shape: {x_test.shape}")
counts = [x_train.shape[0], x_val.shape[0], x_test.shape[0]]
labels = ['Train', 'Val', 'Test']
plt.figure(figsize=(5,4))
plt.bar(labels, counts, color=['blue', 'orange', 'green'])
plt.title('Dataset Split Sizes')
plt.ylabel('Number of Images')
plt.show()

def create_and_train_model(base_model_class, input_shape, num_classes, train_ds, val_ds, epochs, callbacks=None, use_augmentation=True):
    model = Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    if use_augmentation:
        model.add(augmentation)
    base_model = base_model_class(weights='imagenet', include_top=False)
    base_model.trainable = True  
    for layer in base_model.layers[:-10]:
        layer.trainable = False  
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    ) 
    print(f"Training {base_model_class.__name__} model...")
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks
    )
    return model, history

mobilenetv2, history_mobilenetv2 = create_and_train_model(
    MobileNetV2,
    (img_height, img_width, 3),
    num_classes,
    train_ds,
    val_ds,
    epochs,
    callbacks=callbacks,
    use_augmentation=True
)

def plot_training_history_full(history, model_name):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))
    plt.figure(figsize=(14, 5))
    plt.subplot(1,2,1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_training_history_full(history_mobilenetv2, 'MobileNetV2')

print("Evaluating MobileNetV2 model...")
results = mobilenetv2.evaluate(test_ds)
mobilenetv2_loss = results[0]
mobilenetv2_acc = results[1]
print('Test accuracy:', mobilenetv2_acc)
print('Test loss:', mobilenetv2_loss)

y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred_proba = mobilenetv2.predict(test_ds)
y_pred = np.argmax(y_pred_proba, axis=1)

print("\nClassification Report:\n")
print(classification_report(np.argmax(y_true, axis=1), y_pred, digits=4, target_names=class_names))

cm = confusion_matrix(np.argmax(y_true, axis=1), y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

y_true_bin = y_true  
y_score = y_pred_proba
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
plt.figure(figsize=(10, 8))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Class')
plt.legend(loc='lower right')
plt.show()

mis_idx = np.where(np.argmax(y_true, axis=1) != y_pred)[0]
if len(mis_idx) > 0:
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(mis_idx[:15]):
        img = x_test[idx]
        plt.subplot(1, 15, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f'T:{class_names[np.argmax(y_true[idx])]}\nP:{class_names[y_pred[idx]]}')
        plt.axis('off')
    plt.suptitle('First 15 Misclassified Test Images')
    plt.show()
else:
    print('No misclassified images to display.')