from flask import Flask, request, jsonify
from model import predict_flower, accuracy


app = Flask(__name__)


@app.route("/")
def home():
    return jsonify({
        "message": "Flask ML API is running",
        "accuracy": round(accuracy * 100, 2)
    })


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    required_features = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]

    for feature in required_features:
        if feature not in data:
            return jsonify({
                "error": f"Missing feature: {feature}"
            }), 400

    try:

        features = [
            float(data["sepal_length"]),
            float(data["sepal_width"]),
            float(data["petal_length"]),
            float(data["petal_width"])
        ]

        result = predict_flower(features)

        return jsonify({
            "success": True,
            "prediction": result["prediction"],
            "confidence": result["confidence"]
        })

    except ValueError:

        return jsonify({
            "error": "All feature values must be numeric."
        }), 400


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )