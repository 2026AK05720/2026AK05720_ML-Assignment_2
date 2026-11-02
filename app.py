import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)

TARGET = "Target"
MODEL_PATHS = {
    "Logistic Regression": "model/Logistic_Regression.joblib",
    "Decision Tree": "model/Decision_Tree.joblib",
    "KNN": "model/KNN.joblib",
    "Naive Bayes": "model/Naive_Bayes.joblib",
    "SVM": "model/SVM.joblib",
}

st.set_page_config(page_title="Student Outcome Model Lab",
                   page_icon="🎓", layout="wide")
st.title("🎓 Student Dropout & Academic Success: Model Lab")
st.caption("Upload test data, choose a classifier and inspect its performance. "
           "Models are pre-trained; nothing is retrained here.")


@st.cache_resource
def load_assets():
    encoder = joblib.load("model/label_encoder.joblib")
    models = {name: joblib.load(path) for name, path in MODEL_PATHS.items()}
    return encoder, models


def evaluate(model, X, y, n_classes):
    pred = model.predict(X)
    proba = model.predict_proba(X)
    if n_classes == 2:
        auc = roc_auc_score(y, proba[:, 1])
    else:
        auc = roc_auc_score(y, proba, multi_class="ovr", average="weighted")
    scores = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": auc,
        "Precision": precision_score(y, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y, pred, average="weighted"),
        "F1": f1_score(y, pred, average="weighted"),
        "MCC": matthews_corrcoef(y, pred),
    }
    return scores, pred


encoder, models = load_assets()

st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("Upload test CSV", type="csv")
chosen = st.sidebar.selectbox("Choose a model", list(MODEL_PATHS))

if uploaded is None:
    st.info("👈 Upload the test CSV (features + Target column) to begin.")
    st.stop()

data = pd.read_csv(uploaded)
data.columns = data.columns.str.strip()

if TARGET not in data.columns:
    st.error(f"The file must contain a '{TARGET}' column.")
    st.stop()

X = data.drop(columns=[TARGET])
needed = list(models[chosen].feature_names_in_)
missing = [c for c in needed if c not in X.columns]
if missing:
    st.error(f"Missing feature columns: {missing}")
    st.stop()
X = X[needed]
y = encoder.transform(data[TARGET])
n_classes = len(encoder.classes_)

st.write(f"Loaded **{len(data)}** rows and **{X.shape[1]}** features.")
tab1, tab2 = st.tabs(["Single model", "Compare all models"])

with tab1:
    scores, pred = evaluate(models[chosen], X, y, n_classes)
    st.subheader(chosen)
    for col, (name, val) in zip(st.columns(6), scores.items()):
        col.metric(name, f"{val:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(confusion_matrix(y, pred),
                               display_labels=encoder.classes_).plot(
            ax=ax, cmap="Purples", colorbar=False)
        st.pyplot(fig)
    with right:
        st.markdown("**Classification report**")
        st.text(classification_report(y, pred,
                                      target_names=encoder.classes_.astype(str)))

with tab2:
    rows = {}
    for name, mdl in models.items():
        rows[name], _ = evaluate(mdl, X, y, n_classes)
    table = pd.DataFrame(rows).T
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#cdeac0"),
                 use_container_width=True)
    metric_pick = st.selectbox("Plot metric", table.columns, index=0)
    st.bar_chart(table[metric_pick])