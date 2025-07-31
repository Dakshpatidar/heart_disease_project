import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Page setup
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title(" Heart Disease Prediction")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("heart_disease_data.csv")

data = load_data()

# Optional: Show raw data
with st.expander("📊 Show Dataset"):
    st.dataframe(data)

# Input layout
st.markdown("### 🧑‍⚕️ Enter Patient Details")

col1, col2 = st.columns(2)

def get_inputs():
    with col1:
        age = st.slider("Age", 20, 80, 45)
        sex = st.selectbox("Sex", [1, 0])
        cp = st.slider("Chest Pain Type", 0, 3, 1)
        trestbps = st.slider("Resting BP", 90, 200, 120)
        chol = st.slider("Cholesterol", 100, 400, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120", [1, 0])
    with col2:
        restecg = st.slider("Rest ECG", 0, 2, 1)
        thalach = st.slider("Max Heart Rate", 70, 210, 150)
        exang = st.selectbox("Exercise Angina", [1, 0])
        oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
        slope = st.slider("Slope", 0, 2, 1)
        ca = st.slider("Major Vessels (ca)", 0, 4, 0)
        thal = st.slider("Thal", 0, 2, 1)
    return pd.DataFrame({
        'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps], 'chol': [chol],
        'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach], 'exang': [exang],
        'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]
    })

input_df = get_inputs()

# Model training
X = data.drop('target', axis=1)
y = data['target']
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Prediction
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

# Output
st.markdown("### 🩺 Prediction Result")
if prediction == 1:
    st.error(f"🚨 High Risk of Heart Disease\n🧪 Confidence: {probability*100:.2f}%")
else:
    st.success(f"✅ No Heart Disease Detected\n🧪 Confidence: {probability*100:.2f}%")
