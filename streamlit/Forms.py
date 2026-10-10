import streamlit as st

st.set_page_config(
    page_title="My ML Dashbord" ,
    page_icon= "🤖",
    layout="wide"
)

st.title("My Machine Learning Dashbord")
st.header("Welcome!")

st.divider()

with st.form("Customer_form"):
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=50,
        value=12
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=1000.0
    )

    submitted = st.form_submit_button("Submit")


if submitted:
    st.success("Customer details submitted!")
    st.write("Tenure:",tenure)
    st.write("Monthly Charges:",monthly_charges)