import os

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

from tensorflow.keras.datasets import mnist # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.utils import to_categorical # type: ignore

from sklearn.metrics import ( # type: ignore
    confusion_matrix,
    classification_report
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "plots",
    exist_ok=True
)


# ============================================================
# LOAD MNIST DATASET
# ============================================================

print("=" * 70)

print("LOADING MNIST DATASET")

print("=" * 70)


(x_train, y_train), (x_test, y_test) = mnist.load_data()


print("\nTraining images:")
print(x_train.shape)

print("\nTesting images:")
print(x_test.shape)

print("\nTraining labels:")
print(y_train.shape)

print("\nTesting labels:")
print(y_test.shape)


# ============================================================
# DISPLAY SAMPLE IMAGES
# ============================================================

plt.figure(
    figsize=(10, 6)
)

for i in range(10):

    plt.subplot(
        2,
        5,
        i + 1
    )

    plt.imshow(
        x_train[i],
        cmap="gray"
    )

    plt.title(
        f"Digit: {y_train[i]}"
    )

    plt.axis("off")


plt.tight_layout()

plt.savefig(
    "plots/sample_digits.png"
)

plt.close()


# ============================================================
# NORMALIZE PIXEL VALUES
# ============================================================

print("\nNormalizing pixel values...")


x_train = x_train.astype(
    "float32"
) / 255.0


x_test = x_test.astype(
    "float32"
) / 255.0


print(
    "Pixel values are now between 0 and 1."
)


# ============================================================
# ONE-HOT ENCODING
# ============================================================

y_train_encoded = to_categorical(
    y_train,
    10
)

y_test_encoded = to_categorical(
    y_test,
    10
)


# ============================================================
# BUILD NEURAL NETWORK
# ============================================================

print("\nBuilding neural network...")


model = Sequential([

    Flatten(
        input_shape=(28, 28)
    ),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(
        0.2
    ),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(
        0.2
    ),

    Dense(
        10,
        activation="softmax"
    )

])


# ============================================================
# DISPLAY MODEL STRUCTURE
# ============================================================

print("\nMODEL STRUCTURE")

model.summary()


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)

print("TRAINING MODEL")

print("=" * 70)


history = model.fit(

    x_train,

    y_train_encoded,

    epochs=10,

    batch_size=128,

    validation_split=0.1,

    verbose=1

)


# ============================================================
# EVALUATE MODEL
# ============================================================

print("\n" + "=" * 70)

print("EVALUATING MODEL")

print("=" * 70)


test_loss, test_accuracy = model.evaluate(

    x_test,

    y_test_encoded,

    verbose=0

)


print(
    f"\nTest Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)


# ============================================================
# TRAINING ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Training and Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "plots/training_accuracy.png"
)

plt.close()


# ============================================================
# TRAINING LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Training and Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "plots/training_loss.png"
)

plt.close()


# ============================================================
# MAKE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")


predictions = model.predict(
    x_test,
    verbose=0
)


predicted_labels = np.argmax(
    predictions,
    axis=1
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_test,

    predicted_labels

)


plt.figure(
    figsize=(10, 8)
)

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=range(10),

    yticklabels=range(10)

)

plt.title(
    "Confusion Matrix"
)

plt.xlabel(
    "Predicted Digit"
)

plt.ylabel(
    "Actual Digit"
)

plt.tight_layout()

plt.savefig(
    "plots/confusion_matrix.png"
)

plt.close()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)

print("CLASSIFICATION REPORT")

print("=" * 70)


print(
    classification_report(
        y_test,
        predicted_labels
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(
    "models/digit_model.keras"
)


print("\n" + "=" * 70)

print("MODEL TRAINING COMPLETED")

print("=" * 70)


print(
    "\nModel saved at:"
)

print(
    "models/digit_model.keras"
)


print(
    "\nPlots saved in:"
)

print(
    "plots/"
)


print(
    "\nProject completed successfully!"
)