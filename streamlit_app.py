import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import altair as alt

st.set_page_config(
    page_title="AI Project with Babucarr",
    page_icon="🌍",
    layout="wide"
)

st.title("Cardiovascular Risk Prediction")

# ---------------- DATA LOADING ----------------
@st.cache_data
def load_data():
    return pd.read_csv("cvd_synthetic_dataset_v0.2.csv")

df = load_data()

# ---------------- SAFE DATA PREVIEW ----------------
with st.expander("📂 Preview Dataset (first 500 rows only)"):
    st.dataframe(df.head(500), use_container_width=True)

# ---------------- PREPARE DATA ----------------
X = df.drop(columns=["heart_attack_or_stroke_occurred", "patient_id"], errors="ignore")

# Keep only numeric columns
X = X.select_dtypes(include=["int64", "float64"])

# Handle missing values
X = X.fillna(X.median())

y = df["heart_attack_or_stroke_occurred"]

# ---------------- SAFE SCATTER PLOT ----------------
st.subheader("📈 Feature Scatter Plot")

if st.checkbox("Show scatter plot (sampled data)"):
    sample_df = X.sample(min(500, len(X)), random_state=42)
    sample_df["Target"] = y.loc[sample_df.index].astype(str)

    x_feature = st.selectbox("X-axis", X.columns)
    y_feature = st.selectbox("Y-axis", X.columns)

    chart = alt.Chart(sample_df).mark_circle(size=50).encode(
        x=x_feature,
        y=y_feature,
        color="Target:N"
    ).interactive()

    st.altair_chart(chart, width="stretch")

# ---------------- MODEL TRAINING ----------------
st.subheader("🤖 Model Training")

with st.form("train_form"):
    test_size = st.slider("Test set size (%)", 10, 50, 20)
    train_btn = st.form_submit_button("🚀 Train Model")

@st.cache_resource
def train_model(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100, random_state=42
    )

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return model, y_test, preds

if train_btn:
    with st.spinner("Training model..."):
        model, y_test, preds = train_model(X, y, test_size)

    st.success("Training complete")
    st.metric("Accuracy", f"{accuracy_score(y_test, preds):.2f}")

    st.write("Confusion Matrix")
    st.dataframe(confusion_matrix(y_test, preds))

    st.write("Classification Report")
    st.text(classification_report(y_test, preds))

    st.session_state.model = model

# ---------------- PREDICTION ----------------
st.subheader("🔮 Prediction")

if "model" in st.session_state:
    inputs = {
        col: st.number_input(col, float(X[col].mean()))
        for col in X.columns
    }

    if st.button("Predict"):
        user_df = pd.DataFrame([inputs])
        #result = prediction = st.session_state.model.predict(user_df)[0]

if prediction == 1:
    st.error("⚠️ Prediction: HAS heart attack or stroke")
else:
    st.success("✅ Prediction: NO heart attack or stroke")

#else:
    #st.info("Train the model first.")
