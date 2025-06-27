import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dropout, Dense, Input
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 10
PSEUDO_CONFIDENCE_THRESHOLD = 0.95

def preprocess(image, label):
    image = tf.expand_dims(image, -1)
    image = tf.image.grayscale_to_rgb(image)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def create_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=10, train_base=False):
    model = models.Sequential()
    model.add(Input(shape=input_shape))
    base_model = MobileNetV2(weights='imagenet', include_top=False)
    base_model.trainable = train_base
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
    return model

train_csv = pd.read_csv('fashion-mnist/fashion-mnist_train.csv')
test_csv = pd.read_csv('fashion-mnist/fashion-mnist_test.csv')

x_train = train_csv.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.uint8)
y_train = train_csv.iloc[:, 0].values.astype(np.uint8)
x_test = test_csv.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.uint8)
y_test = test_csv.iloc[:, 0].values.astype(np.uint8)

x_train_split, x_val, y_train_split, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

x_labeled = x_train_split[:10000]
y_labeled = y_train_split[:10000]
x_unlabeled = x_train_split[10000:]

labeled_ds = tf.data.Dataset.from_tensor_slices((x_labeled, y_labeled)).map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
unlabeled_ds = tf.data.Dataset.from_tensor_slices((x_unlabeled, np.zeros(len(x_unlabeled)))).map(lambda x, _: (preprocess(x, 0)[0], 0)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

plt.figure(figsize=(15, 3))
indices = random.sample(range(len(x_train)), 15)
for i, idx in enumerate(indices):
    plt.subplot(1, 15, i+1)
    plt.imshow(x_train[idx], cmap='gray')
    plt.title(str(y_train[idx]))
    plt.axis('off')
plt.suptitle('Random Fashion-MNIST Samples')
plt.show()

print(f"Train shape: {x_train_split.shape}, Val shape: {x_val.shape}, Test shape: {x_test.shape}")
counts = [x_train_split.shape[0], x_val.shape[0], x_test.shape[0]]
labels = ['Train', 'Val', 'Test']
plt.figure(figsize=(5,4))
plt.bar(labels, counts, color=['blue', 'orange', 'green'])
plt.title('Dataset Split Sizes')
plt.ylabel('Number of Images')
plt.show()

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.hist(y_train_split, bins=np.arange(11)-0.5, rwidth=0.8)
plt.title('Train Class Balance')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(range(10))
plt.subplot(1,3,2)
plt.hist(y_val, bins=np.arange(11)-0.5, rwidth=0.8)
plt.title('Val Class Balance')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(range(10))
plt.subplot(1,3,3)
plt.hist(y_test, bins=np.arange(11)-0.5, rwidth=0.8)
plt.title('Test Class Balance')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(range(10))
plt.tight_layout()
plt.show()

val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
]

model = create_model(train_base=False)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(labeled_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=callbacks)

preds = model.predict(unlabeled_ds)
pseudo_labels = np.argmax(preds, axis=1)
confidences = np.max(preds, axis=1)
mask = confidences > PSEUDO_CONFIDENCE_THRESHOLD
x_pseudo = x_unlabeled[mask]
y_pseudo = pseudo_labels[mask]

print(f"Pseudo-labeled samples (conf > {PSEUDO_CONFIDENCE_THRESHOLD}): {len(x_pseudo)}")

x_combined = np.concatenate([x_labeled, x_pseudo])
y_combined = np.concatenate([y_labeled, y_pseudo])
combined_ds = tf.data.Dataset.from_tensor_slices((x_combined, y_combined)).map(preprocess).shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

model_finetune = create_model(train_base=True)
model_finetune.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_finetune.fit(combined_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=callbacks)

y_pred = np.argmax(model_finetune.predict(test_ds), axis=1)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, digits=4))

y_pred_proba = model_finetune.predict(test_ds)
y_test_bin = label_binarize(y_test, classes=np.arange(NUM_CLASSES))

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(NUM_CLASSES):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
for i in range(NUM_CLASSES):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Class')
plt.legend(loc='lower right')
plt.show()
