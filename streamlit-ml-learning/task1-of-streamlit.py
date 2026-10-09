import streamlit as st

st.set_page_config(
    page_icon="",
    page_title="ML Learning Hub.",
    layout="wide"
)

st.title("ML Learning Hub")

st.header("My Learning Progress.")

st.write(
    "I Wnat To build A ML Website Which Can Helps " \
    "to Build a Slove Real World Problems."
)

st.warning(
    "Challenge: Balancing ML learning with college."
    "\nBut I love learning ML!"
)

st.divider()

st.subheader("My Current Skills")

st.write("Machine Learning")
st.write("Data Analysis")
st.write("Streamlit Fundamentals")

st.info("This website is a work in progress.")