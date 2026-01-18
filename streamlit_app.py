import streamlit as st
import pandas as pd
import math
from pathlib import Path

# Set the page title and icon
st.set_page_config(
    page_title='Ai project with Babucarr',
    page_icon=':earth_africa:',
)

# Load the dataset
raw_data = pd.read_csv("cvd_synthetic_dataset_v0.2.csv")

# Show the raw data in an expander
with st.expander("View data"):
    st.dataframe(raw_data)

# Display the full dataset outside of expander (optional)
st.write(raw_data)

# Split features and target
st.write("**X (Features)**")
X = raw_data.drop("heart_attack_or_stroke_occurred", axis=1)
st.dataframe(X)

st.write("**Y (Target)**")
Y = raw_data["heart_attack_or_stroke_occurred"]
st.dataframe(Y)
