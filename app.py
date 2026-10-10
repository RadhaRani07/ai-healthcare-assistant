
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="AI Healthcare Assistant",
    page_icon="🩺"
)

@st.cache_resource
def train_model():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)

    X = df.drop("Outcome", axis=1).copy()
    y = df["Outcome"]

    columns_to_clean = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    X[columns_to_clean] = X[columns_to_clean].replace(0, np.nan)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy


st.title("🩺 AI Healthcare Assistant")
st.write(
    "An educational machine learning project "
    "for diabetes outcome classification."
)

st.warning(
    "Educational prototype only. This tool cannot diagnose "
    "diabetes or replace advice from a qualified healthcare professional."
)

try:
    model, accuracy = train_model()
    st.caption(f"Test-set accuracy: {accuracy:.1%}")

    st.subheader("Enter the required values")

    with st.form("prediction_form"):
        pregnancies = st.number_input(
            "Pregnancies", min_value=0, max_value=20, value=1
        )
        glucose = st.number_input(
            "Glucose", min_value=0.0, max_value=300.0, value=120.0
        )
        blood_pressure = st.number_input(
            "Blood Pressure", min_value=0.0, max_value=200.0, value=70.0
        )
        skin_thickness = st.number_input(
            "Skin Thickness", min_value=0.0, max_value=100.0, value=20.0
        )
        insulin = st.number_input(
            "Insulin", min_value=0.0, max_value=900.0, value=80.0
        )
        bmi = st.number_input(
            "BMI", min_value=0.0, max_value=80.0, value=25.0
        )
        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5
        )
        age = st.number_input(
            "Age", min_value=1, max_value=120, value=30
        )

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": diabetes_pedigree,
            "Age": age
        }])

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.info("Model output: Higher Risk Category")
        else:
            st.info("Model output: Lower Risk Category")

        st.caption(
            "This is a model prediction, not a medical diagnosis. "
            "Consult a healthcare professional for medical advice."
        )

except Exception as e:
    st.error(
        "The app could not load the dataset or train the model. "
        "Please try again later."
    )
    st.caption(f"Technical detail: {e}")
