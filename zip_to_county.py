import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def get_county(zip_code: str):
    zip_code = str(zip_code).rjust(5, "0")
    df = pd.read_csv("data/zip_county.csv", dtype=object)
    return df[df["zip"] == zip_code]
