'''import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

import numpy as np

import matplotlib.pyplot as plt

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Prepare the dataset
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    'data/',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    'data/',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Load MobileNetV2 without the top classification layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Build model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    layers.Dense(64),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(train_data, validation_data=val_data, epochs=3)

#record
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)


#Accuracy plot
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Loss plot
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save model
if not os.path.exists("model"):
    os.mkdir("model")
model.save("model/ewaste_model.h5")
print("Model saved at model/ewaste_model.h5")

# Step 8: Evaluate on validation set using F1 Score
val_data.reset()
y_true = val_data.classes

# Get predictions as probabilities
y_pred_prob = model.predict(val_data)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Print metrics
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Not E-Waste', 'E-Waste']))

# Optional: Show F1 separately
f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1:.4f}")

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Create the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot it
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not E-Waste', 'E-Waste'],
            yticklabels=['Not E-Waste', 'E-Waste'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import seaborn as sns

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Data generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    'data/',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

# ✅ Add this line:
print("Class Indices:", train_data.class_indices)


val_data = datagen.flow_from_directory(
    'data/',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    layers.Dense(64),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# Evaluation: F1 and Confusion Matrix
val_data.reset()
y_true = val_data.classes
y_pred_prob = model.predict(val_data)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

print("\n Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Not E-Waste', 'E-Waste']))

f1 = f1_score(y_true, y_pred)
print(f" F1 Score: {f1:.4f}")

#  Confusion Matrix
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not E-Waste', 'E-Waste'],
            yticklabels=['Not E-Waste', 'E-Waste'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()

# Instead of showing here, delay it and batch all plots together

# Accuracy Plot
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Loss Plot
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()

#Show all figures at once
plt.show()

# Save the model
if not os.path.exists("model"):
    os.mkdir("model")
model.save("model/ewaste_model.h5")
print(" Model saved at model/ewaste_model.h5")

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_auc_score
import seaborn as sns
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Data generator with stronger augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    shear_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Training and validation sets
train_data = datagen.flow_from_directory(
    'data/',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    'data/',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Print class indices
print("Class Indices:", train_data.class_indices)

# Compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# Load MobileNetV2 and unfreeze top layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Build model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    layers.Dense(64),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    class_weight=class_weights,
    callbacks=[early_stop]
)

# Evaluation
val_data.reset()
y_true = val_data.classes
y_pred_prob = model.predict(val_data)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['E-WASTE', 'NON-E-WASTE']))

f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1:.4f}")

auc = roc_auc_score(y_true, y_pred_prob)
print(f"AUC: {auc:.4f}")

# Confusion matrix
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['E-WASTE', 'NON-E-WASTE'],
            yticklabels=['E-WASTE', 'NON-E-WASTE'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()

# Accuracy plot
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Loss plot
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

# Save model
if not os.path.exists("model"):
    os.mkdir("model")
model.save("model/ewaste_model.h5")
print("Model saved at model/ewaste_model.h5")
'''
'''
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_auc_score
import seaborn as sns
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Data generators with augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    shear_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    'Data/',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    color_mode='grayscale'
)

val_data = datagen.flow_from_directory(
    'Data/',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    color_mode='grayscale'
)

# Class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights))

# Replicate grayscale to 3 channels
inputs = tf.keras.Input(shape=(224, 224, 1))
x = layers.Concatenate()([inputs, inputs, inputs])

# Load EfficientNetB3 on replicated grayscale input
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_tensor=x)
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

# Model architecture
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(128)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=40,
    class_weight=class_weights,
    callbacks=[early_stop, lr_schedule]
)   

# Evaluation
y_true = val_data.classes
y_pred_prob = model.predict(val_data)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['E-WASTE', 'NON-E-WASTE']))

f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1:.4f}")

auc = roc_auc_score(y_true, y_pred_prob)
print(f"AUC: {auc:.4f}")

# Confusion matrix
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['E-WASTE', 'NON-E-WASTE'],
            yticklabels=['E-WASTE', 'NON-E-WASTE'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()

# Accuracy plot
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Loss plot
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

# Save model
if not os.path.exists("Model"):
    os.mkdir("Model")
model.save("Model/ewaste_model_efficientnet.h5")
print("Model saved at Model/ewaste_model_efficientnet.h5")
'''
'''
# Train_Model.py

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# Global paths
DATA_DIR = os.path.join("Data")
MODEL_PATH = os.path.join("Model", "ewaste_model.h5")

def load_and_preprocess_data(img_size=(224, 224), batch_size=32, val_split=0.2):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=val_split,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=val_split,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )

    # Prefetching for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds

def build_model(input_shape=(224, 224, 3)):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the base

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def plot_training_history(history):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    img_size = (224, 224)
    batch_size = 32

    train_ds, val_ds = load_and_preprocess_data(img_size=img_size, batch_size=batch_size)

    model = build_model(input_shape=img_size + (3,))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_ds, validation_data=val_ds, epochs=10)

    model.save(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    plot_training_history(history)

if __name__ == "__main__":
    main()

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, f1_score, roc_auc_score
)

# Force matplotlib to work in script environments like VS Code
import matplotlib
matplotlib.use('Agg')

# Paths
DATA_DIR = os.path.join("Data")
MODEL_PATH = os.path.join("Model", "ewaste_model.h5")
CONFUSION_MATRIX_PATH = os.path.join("Model", "confusion_matrix.png")
ACCURACY_LOSS_PLOT_PATH = os.path.join("Model", "training_plot.png")
CLASS_REPORT_PATH = os.path.join("Model", "classification_report.txt")

def load_and_preprocess_data(img_size=(128, 128), batch_size=32, val_split=0.2):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=val_split,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=val_split,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )

    AUTOTUNE = tf.data.AUTOTUNE
    return train_ds.prefetch(AUTOTUNE), val_ds.prefetch(AUTOTUNE)

def build_model(input_shape=(128, 128, 3)):
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))  # Dropout after 1st block

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))  # Dropout after 2nd block

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))  # Dropout before final output

    model.add(layers.Dense(1, activation='sigmoid'))

    return model

def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(ACCURACY_LOSS_PLOT_PATH)
    print(f"Training plot saved to '{ACCURACY_LOSS_PLOT_PATH}'")

def evaluate_and_report(model, val_ds, class_names):
    y_true, y_pred, y_prob = [], [], []

    print("\n[INFO] Evaluating model...\n")

    for images, labels in val_ds:
        probs = model.predict(images, verbose=0)
        preds = (probs > 0.5).astype("int32").flatten()
        y_true.extend(labels.numpy())
        y_pred.extend(preds)
        y_prob.extend(probs.flatten())

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    print(f"Confusion matrix saved to '{CONFUSION_MATRIX_PATH}'")

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=[n.upper() for n in class_names])
    with open(CLASS_REPORT_PATH, "w") as f:
        f.write("Classification Report\n")
        f.write("=====================\n")
        f.write(report)
    print(f"Classification report saved to '{CLASS_REPORT_PATH}'")

    # F1 Score and AUC
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    print(f"\nF1 Score: {f1:.4f}")
    print(f"AUC Score: {auc:.4f}")

def main():
    img_size = (128, 128)
    batch_size = 32
    class_names = ['e-waste', 'non-e-waste']

    train_ds, val_ds = load_and_preprocess_data(img_size=img_size, batch_size=batch_size)

    model = build_model(input_shape=img_size + (3,))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_ds, validation_data=val_ds, epochs=10)

    model.save(MODEL_PATH)
    print(f"Model saved to '{MODEL_PATH}'")

    plot_training_history(history)
    evaluate_and_report(model, val_ds, class_names)

if __name__ == "__main__":
    main()

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, f1_score, roc_auc_score
)

import matplotlib
matplotlib.use('Agg')  # For VS Code or headless environments

# Paths
DATA_DIR = os.path.join("Data")
MODEL_PATH = os.path.join("Model", "ewaste_model.h5")
CONFUSION_MATRIX_PATH = os.path.join("Model", "confusion_matrix.png")
ACCURACY_LOSS_PLOT_PATH = os.path.join("Model", "training_plot.png")
CLASS_REPORT_PATH = os.path.join("Model", "classification_report.txt")

def load_and_preprocess_data(img_size=(128, 128), batch_size=32, val_split=0.2):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=val_split,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=val_split,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )

    AUTOTUNE = tf.data.AUTOTUNE
    return train_ds.prefetch(AUTOTUNE), val_ds.prefetch(AUTOTUNE)

def build_model(input_shape=(128, 128, 3)):
    model = models.Sequential([
        # Data Augmentation
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),

        # Normalization
        layers.Rescaling(1./255),

        # CNN Blocks + Dropout
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),

        layers.Dense(1, activation='sigmoid')
    ])
    return model

def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(ACCURACY_LOSS_PLOT_PATH)
    print(f"Training plot saved to '{ACCURACY_LOSS_PLOT_PATH}'")

def evaluate_and_report(model, val_ds, class_names):
    y_true, y_pred, y_prob = [], [], []

    print("\n[INFO] Evaluating model...\n")

    for images, labels in val_ds:
        probs = model.predict(images, verbose=0)
        preds = (probs > 0.5).astype("int32").flatten()
        y_true.extend(labels.numpy())
        y_pred.extend(preds)
        y_prob.extend(probs.flatten())

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    print(f"Confusion matrix saved to '{CONFUSION_MATRIX_PATH}'")

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=[n.upper() for n in class_names])
    with open(CLASS_REPORT_PATH, "w") as f:
        f.write("Classification Report\n")
        f.write("=====================\n")
        f.write(report)
    print(f"Classification report saved to '{CLASS_REPORT_PATH}'")

    # F1 Score and AUC
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    print(f"\nF1 Score: {f1:.4f}")
    print(f"AUC Score: {auc:.4f}")

def main():
    img_size = (128, 128)
    batch_size = 32
    class_names = ['e-waste', 'non-e-waste']

    train_ds, val_ds = load_and_preprocess_data(img_size=img_size, batch_size=batch_size)

    model = build_model(input_shape=img_size + (3,))

    # Lower learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Early stopping to prevent overfitting
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[early_stop]
    )

    model.save(MODEL_PATH)
    print(f"Model saved to '{MODEL_PATH}'")

    plot_training_history(history)
    evaluate_and_report(model, val_ds, class_names)

if __name__ == "__main__":
    main()

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, f1_score, roc_auc_score
)

import matplotlib
matplotlib.use('Agg')  # For VS Code or headless environments

# Paths
DATA_DIR = os.path.join("Data")
MODEL_PATH = os.path.join("Model", "ewaste_model_alexnet.h5")
CONFUSION_MATRIX_PATH = os.path.join("Model", "confusion_matrix.png")
ACCURACY_LOSS_PLOT_PATH = os.path.join("Model", "training_plot.png")
CLASS_REPORT_PATH = os.path.join("Model", "classification_report.txt")

def load_and_preprocess_data(img_size=(227, 227), batch_size=32, val_split=0.2):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=val_split,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=val_split,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )

    AUTOTUNE = tf.data.AUTOTUNE
    return train_ds.prefetch(AUTOTUNE), val_ds.prefetch(AUTOTUNE)

def build_alexnet(input_shape=(227, 227, 3)):
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=input_shape),
        layers.Conv2D(96, (11, 11), strides=4, activation='relu'),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Conv2D(256, (5, 5), padding='same', activation='relu'),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(ACCURACY_LOSS_PLOT_PATH)
    print(f"Training plot saved to '{ACCURACY_LOSS_PLOT_PATH}'")

def evaluate_and_report(model, val_ds, class_names):
    y_true, y_pred, y_prob = [], [], []

    print("\n[INFO] Evaluating model...\n")

    for images, labels in val_ds:
        probs = model.predict(images, verbose=0)
        preds = (probs > 0.5).astype("int32").flatten()
        y_true.extend(labels.numpy())
        y_pred.extend(preds)
        y_prob.extend(probs.flatten())

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    print(f"Confusion matrix saved to '{CONFUSION_MATRIX_PATH}'")

    report = classification_report(y_true, y_pred, target_names=[n.upper() for n in class_names])
    with open(CLASS_REPORT_PATH, "w") as f:
        f.write("Classification Report\n")
        f.write("=====================\n")
        f.write(report)
    print(f"Classification report saved to '{CLASS_REPORT_PATH}'")

    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    print(f"\nF1 Score: {f1:.4f}")
    print(f"AUC Score: {auc:.4f}")

def main():
    img_size = (227, 227)
    batch_size = 32
    class_names = ['e-waste', 'non-e-waste']

    train_ds, val_ds = load_and_preprocess_data(img_size=img_size, batch_size=batch_size)

    model = build_alexnet(input_shape=img_size + (3,))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[early_stop]
    )

    model.save(MODEL_PATH)
    print(f"Model saved to '{MODEL_PATH}'")

    plot_training_history(history)
    evaluate_and_report(model, val_ds, class_names)

if __name__ == "__main__":
    main()

import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Paths
DATA_DIR = os.path.join("Data")
MODEL_PATH = os.path.join("Model", "ewaste_model_alexnet.h5")
IMG_SIZE = (227, 227)
BATCH_SIZE = 8
CLASS_NAMES = ['e-waste', 'non-e-waste']

# Load and preprocess data
def load_data():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )

    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE)

# Build AlexNet-style model with 10 Dense layers
def build_model():
    model = models.Sequential()

    # First conv layers like AlexNet
    model.add(layers.Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=(227, 227, 3)))
    model.add(layers.MaxPooling2D((3, 3), strides=2))
    model.add(layers.Conv2D(256, (5, 5), padding="same", activation='relu'))
    model.add(layers.MaxPooling2D((3, 3), strides=2))
    model.add(layers.Conv2D(384, (3, 3), padding="same", activation='relu'))
    model.add(layers.Conv2D(384, (3, 3), padding="same", activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), padding="same", activation='relu'))
    model.add(layers.MaxPooling2D((3, 3), strides=2))

    model.add(layers.Flatten())

    # 10 Dense layers
    for _ in range(5):
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.3))

    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def main():
    train_ds, val_ds = load_data()
    model = build_model()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=[early_stop])

    model.save(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
'''
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score, roc_auc_score
import numpy as np

# Paths
DATA_DIR = os.path.join("Data")
MODEL_PATH = os.path.join("Model", "ewaste_model_alexnet.h5")
IMG_SIZE = (227, 227)
BATCH_SIZE = 8
CLASS_NAMES = ['e-waste', 'non-e-waste']
PLOT_DIR = "Model"
TRAINING_PLOT_PATH = os.path.join(PLOT_DIR, "training_plot.png")
CONFUSION_MATRIX_PATH = os.path.join(PLOT_DIR, "confusion_matrix.png")

# Load and preprocess data
def load_data():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )

    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE)

# Build full AlexNet model with 5 extra Dense layers + Dropout
def build_model():
    model = models.Sequential()

    # AlexNet-style convolutional layers
    model.add(layers.Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=(227, 227, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))

    model.add(layers.Conv2D(256, (5, 5), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))

    model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))

    model.add(layers.Flatten())

    # Additional 5 dense layers with dropout
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.3))

    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

# Plot accuracy and loss
def plot_history(history):
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(TRAINING_PLOT_PATH)
    print(f"[INFO] Training plot saved to: {TRAINING_PLOT_PATH}")

# Evaluate model and generate confusion matrix + metrics
def evaluate_model(model, val_ds):
    y_true, y_pred, y_prob = [], [], []

    for images, labels in val_ds:
        probs = model.predict(images, verbose=0).flatten()
        preds = (probs > 0.5).astype("int32")
        y_true.extend(labels.numpy())
        y_pred.extend(preds)
        y_prob.extend(probs)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[n.upper() for n in CLASS_NAMES])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    print(f"[INFO] Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")

    # Classification Report
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # F1 Score and AUC
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC Score: {auc:.4f}")

def main():
    train_ds, val_ds = load_data()
    model = build_model()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=[early_stop])

    model.save(MODEL_PATH)
    print(f"[INFO] Model saved to: {MODEL_PATH}")

    plot_history(history)
    evaluate_model(model, val_ds)

if __name__ == "__main__":
    main()

