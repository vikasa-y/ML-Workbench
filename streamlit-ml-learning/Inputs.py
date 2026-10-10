import streamlit as st

st.set_page_config(
    page_title="My ML Dashbord" ,
    page_icon= "🤖",
    layout="wide"
)

st.title("My Machine Learning Dashbord")
st.header("Welcome!")

st.divider()

terure = st.slider(
    "Customer tenure (Months)",
    min_value=0,
    max_value=42,
    value=12
)

st.write("Terune:", terure)

st.divider()


monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=100.0,
    value=70.0,
    step=5.0
)

st.write("Monthly charges:", monthly_charges)


show_details = st.checkbox("Show additional details")

if show_details:
    st.write("Here are the additional details.")

st.divider()

show_chart = st.toggle("Show chart", value=True)

if show_chart:
    st.line_chart([1,3,5,10,15,12,8,20,25])


st.divider()

date = st.date_input("Select a date")
time = st.time_input("Select a time")

st.write("Date:", date)
st.write("Time:", time)