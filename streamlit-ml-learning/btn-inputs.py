import streamlit as st

st.set_page_config(
    page_title="My ML Dashbord" ,
    page_icon= "🤖",
    layout="wide"
)

st.title("My Machine Learning Dashbord")
st.header("Welcome!")

st.divider()
name = st.text_input("Enter your name")

if name:
    st.success(f"Welcome to my ML Dashboard, {name}!")

  
if st.button("Show message"):
    st.success("you clicked this button!")

st.divider()

model = st.selectbox(
    "Choose a Model",
    ["Logistic Regression" , "Decision Tree" , "Random Forest"]
)

if st.button("Show my current learning"):
    st.info("I am learning Streamlit to build " \
    "interactive Machine Learning dashboards.")