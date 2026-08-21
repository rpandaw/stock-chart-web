import streamlit as st

st.title("My Stock Chart")

symbol = st.text_input("Enter Stock Symbol", "MRVL")

if st.button("Chart"):
    st.success("You selected: " + symbol)
