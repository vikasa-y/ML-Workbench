import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression



X = np.array([[1] ,[2] ,[3] ,[4] ,[5] ,[6]])
y = np.array([0,0,0,1,1,1])

model = LogisticRegression()

model.fit(X,y)

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}

.card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 25px;
    border: 1px solid rgba(255,255,255,0.15);
}

</style>
""", unsafe_allow_html=True)



st.markdown("""
<div class="card">

<h2>Student Pass Prediction</h2>

<p>Enter Students Hours to find Pass Predict.</p>

</div>
""", unsafe_allow_html=True)

st.title("🎓 Student Pass Prediction")

hours = st.number_input(
    "Enter Study Hours:",
    min_value=0.0,
    max_value=15.0,
    value=0.0
)

if st.button("Predict"):
    x_new = np.array([[hours]])

    prediction = model.predict(x_new)
    probability = model.predict_proba(x_new)

    pass_probability = probability[0][1]

    if prediction[0] == 1:
        st.success("Student is Likely to PASS")
    else:
        st.error("Student is Likely to FAIL")

    st.write(f"Pass Probability: {pass_probability *100:.2f}%")

    st.progress(float(pass_probability))