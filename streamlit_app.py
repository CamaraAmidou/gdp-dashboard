import streamlit as st
import pandas as pd
import math
from pathlib import Path

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Ai project with Babucarr',
    page_icon=':earth_africa:',
)
raw_data = pd.read_csv("cvd_synthetic_dataset_v0.2.csv")
st.write(raw_data)

with st.expander("View data"):
    
    df.dataframe(df)



    
