
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

