import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# Style
st.markdown("""
    <style>
    body {
        background-color: #f0f4f8;
    }
    .section-title {
        background-color: #0d47a1;
        color: #bbdefb;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-size: 1.8rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    label {
        color: #00BFFF !important;  /* Light Blue color for visibility */
        font-size: 1.2rem !important;
        font-weight: 600 !important;
    }
    .stSlider > div, .stSelectbox > div {
        font-size: 1.2rem !important;
    }
    .stButton>button {
        background-color: #1565c0;
        color: white;
        font-weight: bold;
        padding: 0.6rem 2.5rem;
        font-size: 1rem;
        border-radius: 8px;
        margin: 1rem auto;
        display: block;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0d47a1;
        transform: scale(1.03);
    }
    .prediction-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 2.5rem;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
        border-left: 8px solid #1565c0;
    }
    .prediction-card h4 {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .prediction-card p {
        font-size: 1.1rem;
        margin: 0.2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Heading
st.markdown('<div class="section-title">💓 Heart Disease Prediction</div>', unsafe_allow_html=True)

# Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv("heart_disease_data.csv")

data = load_data()

# Input Form
st.markdown("### 🧑‍⚕️ Enter Patient Details Below")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 80, 45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.slider("Chest Pain Type (0-3)", 0, 3, 1)
    trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)
    chol = st.slider("Cholesterol", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120", ["Yes", "No"])

with col2:
    restecg = st.slider("Rest ECG (0-2)", 0, 2, 1)
    thalach = st.slider("Max Heart Rate", 70, 210, 150)
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.slider("Slope (0-2)", 0, 2, 1)
    ca = st.slider("Major Vessels Colored (ca)", 0, 4, 0)
    thal = st.slider("Thalassemia (0-2)", 0, 2, 1)

# Convert to numeric
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

input_df = pd.DataFrame({
    'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps],
    'chol': [chol], 'fbs': [fbs], 'restecg': [restecg],
    'thalach': [thalach], 'exang': [exang], 'oldpeak': [oldpeak],
    'slope': [slope], 'ca': [ca], 'thal': [thal]
})

# Train Model
X = data.drop('target', axis=1)
y = data['target']
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Predict Button placed just below form
predict_clicked = st.button("🔍 Predict")

# Prediction Output
if predict_clicked:
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.markdown(f"""
            <div class="prediction-card">
                <h4 style='color:#c62828;'>🚨 High Risk of Heart Disease</h4>
                <p><strong>Confidence Level:</strong> {probability*100:.2f}%</p>
                <p><strong>Action:</strong> Please consult a cardiologist as soon as possible.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="prediction-card">
                <h4 style='color:#2e7d32;'>✅ No Heart Disease Detected</h4>
                <p><strong>Confidence Level:</strong> {probability*100:.2f}%</p>
                <p><strong>Suggestion:</strong> Maintain a healthy lifestyle to keep your heart strong.</p>
            </div>
        """, unsafe_allow_html=True)
