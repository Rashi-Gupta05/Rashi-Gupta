import os

# pyright: reportMissingImports=false
import numpy as np

from flask import (
    Flask,
    render_template,
    request
)

from tensorflow.keras.models import load_model

from PIL import Image


# ============================================================
# CREATE FLASK APPLICATION
# ============================================================

app = Flask(
    __name__
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = load_model(
    "models/digit_model.keras"
)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------------------
        # Check uploaded file
        # ----------------------------------------------------

        if "digit_image" not in request.files:

            return render_template(
                "index.html",
                error="Please upload an image."
            )


        file = request.files[
            "digit_image"
        ]


        if file.filename == "":

            return render_template(
                "index.html",
                error="Please select an image."
            )


        # ----------------------------------------------------
        # Open image
        # ----------------------------------------------------

        image = Image.open(
            file.stream
        )


        # ----------------------------------------------------
        # Convert to grayscale
        # ----------------------------------------------------

        image = image.convert(
            "L"
        )


        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        image = image.resize(
            (28, 28)
        )


        # ----------------------------------------------------
        # Convert to NumPy
        # ----------------------------------------------------

        image_array = np.array(
            image
        ).astype(
            "float32"
        )


        # ----------------------------------------------------
        # Invert if necessary
        # ----------------------------------------------------

        if image_array.mean() > 127:

            image_array = (
                255 - image_array
            )


        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        image_array = (
            image_array / 255.0
        )


        # ----------------------------------------------------
        # Add batch dimension
        # ----------------------------------------------------

        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        prediction = model.predict(
            image_array,
            verbose=0
        )


        predicted_digit = int(
            np.argmax(
                prediction[0]
            )
        )


        confidence = float(
            np.max(
                prediction[0]
            ) * 100
        )


        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return render_template(

            "index.html",

            prediction=predicted_digit,

            confidence=f"{confidence:.2f}"

        )


    except Exception as e:

        return render_template(

            "index.html",

            error=f"Error: {str(e)}"

        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )