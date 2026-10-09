import streamlit as st

st.set_page_config(
    page_title="My ML Dashbord" ,
    page_icon= "🤖",
    layout="wide"
)

st.title("My Machine Learning Dashbord")
st.header("Welcome!")

st.write(
    "I am Learning To Build End-to-End " \
    "Machine Learning Applications."
)

st.markdown("**My Goal:** Learn, Build, test, and improve.")

st.info("This is MY frist Streamlit Application.")

st.success("Application Setup Completed!")