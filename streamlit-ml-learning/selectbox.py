import streamlit as st

st.set_page_config(
    page_title="My ML Dashbord" ,
    page_icon= "🤖",
    layout="wide"
)

st.title("My Machine Learning Dashbord")
st.header("Welcome!")

model = st.selectbox(
    "Choose a Machine Learning model",
    ["Logistic Regression" , "Decision Tree", "Random Forest"]
)


st.write("Selected model :",model)

st.divider()

level = st.radio(
    "choose your level",
    ["Beginner", "Intermediate" , "Advanced"],
    horizontal=False
)

st.write("You Selected :", level)

st.divider()

metrics = st.multiselect(
    "Choose evaluation metrics",
    ["Accuracy", "Precision", "Recall", "F1-score"],
    default=["Accuracy", "F1-score"]
)

st.write("Selected metrics:", metrics)