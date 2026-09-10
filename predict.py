import sys

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from PIL import Image


# ============================================================
# CHECK COMMAND LINE ARGUMENT
# ============================================================

if len(sys.argv) < 2:

    print(
        "\nPlease provide an image path."
    )

    print(
        "\nExample:"
    )

    print(
        "python predict.py digit.png"
    )

    sys.exit()


image_path = sys.argv[1]


# ============================================================
# LOAD MODEL
# ============================================================

model = load_model(
    "models/digit_model.keras"
)


# ============================================================
# LOAD IMAGE
# ============================================================

try:

    image = Image.open(
        image_path
    )

except Exception as e:

    print(
        f"Unable to open image: {e}"
    )

    sys.exit()


# ============================================================
# CONVERT IMAGE TO GRAYSCALE
# ============================================================

image = image.convert(
    "L"
)


# ============================================================
# RESIZE IMAGE
# ============================================================

image = image.resize(
    (28, 28)
)


# ============================================================
# CONVERT TO NUMPY ARRAY
# ============================================================

image_array = np.array(
    image
).astype(
    "float32"
)


# ============================================================
# INVERT IMAGE IF NECESSARY
# ============================================================

# MNIST generally uses:
# black background + white digit

if image_array.mean() > 127:

    image_array = 255 - image_array


# ============================================================
# NORMALIZE
# ============================================================

image_array = image_array / 255.0


# ============================================================
# ADD BATCH DIMENSION
# ============================================================

image_array = np.expand_dims(
    image_array,
    axis=0
)


# ============================================================
# PREDICT
# ============================================================

prediction = model.predict(
    image_array,
    verbose=0
)


predicted_digit = np.argmax(
    prediction[0]
)


confidence = np.max(
    prediction[0]
) * 100


# ============================================================
# DISPLAY RESULT
# ============================================================

print("=" * 60)

print("HANDWRITTEN DIGIT PREDICTION")

print("=" * 60)

print(
    f"\nPredicted Digit: {predicted_digit}"
)

print(
    f"Confidence: {confidence:.2f}%"
)


# ============================================================
# DISPLAY IMAGE
# ============================================================

plt.imshow(
    image_array[0],
    cmap="gray"
)

plt.title(
    f"Predicted Digit: {predicted_digit}"
)

plt.axis(
    "off"
)

plt.show()