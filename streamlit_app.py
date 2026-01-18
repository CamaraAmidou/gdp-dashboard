import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import altair as alt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Project with Babucarr",
    page_icon="🌍",
    layout="wide"
)

st.title("Cardiovascular Risk Prediction App")

# ---------------- CACHE DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("cvd_synthetic_dataset_v0.2.csv")

raw_data = load_data()

# ---------------- VIEW DATA ----------------
with st.expander("View Raw Dataset"):
    st.dataframe(raw_data, use_container_width=True)

# ---------------- PREPARE DATA ----------------
X = raw_data.drop(
    columns=["heart_attack_or_stroke_occurred", "patient_id"],
    errors="ignore"
)
Y = raw_data["heart_attack_or_stroke_occurred"]

with st.expander("View Features (X)"):
    st.dataframe(X, use_container_width=True)

with st.expander("View Target (Y)"):
    st.dataframe(Y, use_container_width=True)

# ---------------- SCATTER PLOT ----------------
st.subheader("Feature Scatter Plot")

feature_options = X.columns.tolist()
x_feature = st.selectbox("Select X-axis feature", feature_options)
y_feature = st.selectbox("Select Y-axis feature", feature_options)

plot_data = X.copy()
plot_data["Target"] = Y.astype(str)

scatter = alt.Chart(plot_data).mark_circle(size=60).encode(
    x=x_feature,
    y=y_feature,
    color="Target:N",
    tooltip=list(plot_data.columns)
).interactive()

st.altair_chart(scatter, width="stretch")

# ---------------- MODEL TRAINING ----------------
st.subheader("Gradient Boosting Model Training")

test_size = st.slider("Test set size (%)", 10, 50, 20)

@st.cache_resource
def train_model(X, Y, test_size):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size / 100, random_state=42
    )

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    return model, X_test, Y_test, Y_pred

if st.button("Train Model"):
    with st.spinner("Training model... Please wait"):
        model, X_test, Y_test, Y_pred = train_model(X, Y, test_size)

    accuracy = accuracy_score(Y_test, Y_pred)
    st.success("Model trained successfully!")

    st.metric("Accuracy", f"{accuracy:.2f}")

    st.write("### Confusion Matrix")
    st.dataframe(confusion_matrix(Y_test, Y_pred))

    st.write("### Classification Report")
    st.text(classification_report(Y_test, Y_pred))

    st.session_state["model"] = model

# ---------------- PREDICTION ----------------
st.subheader("Make a Prediction")

if "model" not in st.session_state:
    st.info("Train the model first to enable predictions.")
else:
    user_input = {}
    for col in X.columns:
        user_input[col] = st.number_input(
            f"{col}",
            value=float(X[col].mean())
        )

    if st.button("Predict"):
        user_df = pd.DataFrame([user_input])
        prediction = st.session_state["model"].predict(user_df)[0]
        st.success(f"Predicted Heart Attack or Stroke: **{prediction}**")
