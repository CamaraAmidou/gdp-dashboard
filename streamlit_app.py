import streamlit as st
import pandas as pd
import math
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import altair as alt

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

# Display the full dataset outside of expander 
#st.write(raw_data)

# Split features and target
X = raw_data.drop("heart_attack_or_stroke_occurred", axis=1)
Y = raw_data["heart_attack_or_stroke_occurred"]

# Show features in an expander
with st.expander("View Features (X)"):
    st.dataframe(X)

# Show target in an expander
with st.expander("View Target (Y)"):
    st.dataframe(Y)
    
    # Scatter plot section
st.write("## Scatter Plot of Features")

# Let user select features for x and y axes
feature_options = X.columns.tolist()
x_feature = st.selectbox("Select X-axis feature", feature_options)
y_feature = st.selectbox("Select Y-axis feature", feature_options)

# Combine X and Y for plotting
plot_data = X.copy()
plot_data['Target'] = Y

# Create scatter plot
scatter = alt.Chart(plot_data).mark_circle(size=60).encode(
    x=x_feature,
    y=y_feature,
    color='Target:N',  # Treat target as categorical
    tooltip=list(plot_data.columns)
).interactive()

st.altair_chart(scatter, use_container_width=True)



