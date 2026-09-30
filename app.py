import streamlit as st
import requests


API_URL = "http://127.0.0.1:5000/predict"


st.set_page_config(
    page_title="Iris Prediction",
    page_icon="🌸",
    layout="centered"
)


st.title("🌸 Iris Flower Prediction")
st.write("Streamlit Frontend + Flask API Backend")


st.subheader("Enter Flower Measurements")


sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=5.1,
    step=0.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=3.5,
    step=0.1
)

petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=1.4,
    step=0.1
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=0.2,
    step=0.1
)


if st.button("Predict Flower"):

    data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }

    try:

        response = requests.post(
            API_URL,
            json=data,
            timeout=10
        )

        if response.status_code == 200:

            result = response.json()

            st.success("Prediction generated successfully!")

            st.subheader("Prediction Result")

            st.write(
                f"**Flower:** {result['prediction'].title()}"
            )

            st.write(
                f"**Confidence:** "
                f"{result['confidence'] * 100:.2f}%"
            )

        else:

            st.error(
                f"API Error: {response.text}"
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "Could not connect to Flask API. "
            "Please make sure the Flask server is running."
        )

    except requests.exceptions.Timeout:

        st.error(
            "The Flask API request timed out."
        )