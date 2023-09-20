import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def get_features():
    features = pd.read_csv(
        "./data/feature_info.csv",
        dtype={"bands": "Int64", "color_scale_rev": bool},
    )

    def str_to_list(s):
        if pd.isna(s):
            return None
        elif "|" in s:
            return s.split("|")
        else:
            return s.split(",")

    def str_list_to_int_list(str_list):
        if str_list is not None:
            return [int(s) for s in str_list]
        else:
            return None

    # apply the function to the "ticks" column
    features["ticks"] = features["ticks"].apply(str_to_list).apply(str_list_to_int_list)
    features["options"] = features["options"].apply(str_to_list)

    return features
