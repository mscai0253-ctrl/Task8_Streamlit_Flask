from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create ML pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=200))
])


# Train model
model.fit(X_train, y_train)


# Calculate accuracy
accuracy = model.score(X_test, y_test)


# Class names
class_names = iris.target_names


def predict_flower(features):
    prediction = model.predict([features])[0]

    probability = model.predict_proba([features]).max()

    return {
        "prediction": class_names[prediction],
        "confidence": float(probability)
    }