from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
TARGET = "Target"
BITS_ID = "2026AK05720"
APP_TITLE = "Student Outcome Predictor"
DATASET_NAME = "UCI Predict Students' Dropout and Academic Success"
MAX_ROWS = 20_000
BASE_DIR = Path(__file__).parent

MODEL_FILES = {
    "Logistic Regression": "model/Logistic_Regression.joblib",
    "Decision Tree": "model/Decision_Tree.joblib",
    "KNN": "model/KNN.joblib",
    "Naive Bayes": "model/Naive_Bayes.joblib",
    "SVM": "model/SVM.joblib",
}
SHORT = {"Logistic Regression": "LR", "Decision Tree": "DT", "KNN": "KNN",
         "Naive Bayes": "NB", "SVM": "SVM"}
DEFINITIONS = {
    "Logistic Regression": "A linear model that estimates the probability of each outcome from a weighted sum of the student's features.",
    "Decision Tree": "A model that follows a sequence of learned if-then rules on the features to decide which outcome a student belongs to.",
    "KNN": "A model that compares a student with the 7 most similar students in the training data and takes the majority outcome.",
    "Naive Bayes": "A probabilistic model that combines how likely each feature value is under each outcome, assuming features are independent.",
    "SVM": "A model that finds the boundary with the widest margin between outcomes, using an RBF kernel for curved boundaries.",
}
METRIC_INFO = {
    "Accuracy": ("Share of all students classified correctly", "0-1"),
    "AUC": ("How well the model ranks outcomes (one-vs-rest)", "0-1"),
    "Precision": ("How often a predicted outcome is right", "0-1"),
    "Recall": ("How many actual cases of each outcome are found", "0-1"),
    "F1": ("Balance between precision and recall", "0-1"),
    "MCC": ("Correlation between prediction and truth", "-1 to 1"),
}

# Palette: change these to re-theme the whole app.
INK = "#14213D"        # sidebar and hero background
INK_SOFT = "#1E3157"   # cards inside the sidebar
ACCENT = "#FCA311"     # buttons and highlights
ACCENT_DARK = "#B36F00"
PAPER = "#F6F7FB"      # page background
CARD = "#FFFFFF"
TEXT = "#14213D"
MUTED = "#5B6784"

st.set_page_config(page_title=APP_TITLE, page_icon="🎓", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Plus Jakarta Sans', sans-serif; }}
.stApp {{ background: {PAPER}; color: {TEXT}; }}
.block-container {{ padding-top: 2.2rem; max-width: 1280px; }}

/* sidebar */  
[data-testid="stSidebar"] {{ background: {INK}; }}
[data-testid="stSidebar"] * {{ color: #E7ECF7; }}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {{
    background: {INK_SOFT}; border: 1px solid rgba(255,255,255,.12); border-radius: 16px; }}
[data-testid="stSidebar"] [data-baseweb="select"] > div {{ background: #FFFFFF; }}
[data-testid="stSidebar"] [data-baseweb="select"] * {{ color: {INK} !important; }}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
    background: rgba(255,255,255,.04); border: 1px dashed rgba(255,255,255,.4); border-radius: 12px; }}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {{
    background: transparent; border: 1px solid rgba(255,255,255,.5); }}
[data-testid="stSidebar"] .stDownloadButton button {{
    background: {ACCENT}; border: none; border-radius: 10px; width: 100%; font-weight: 700; }}
[data-testid="stSidebar"] .stDownloadButton button * {{ color: {INK} !important; }}
[data-testid="stSidebar"] [data-testid="stExpander"] {{
    background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.12); border-radius: 12px; }}
[data-testid="stSidebar"] [data-testid="stCode"],
[data-testid="stSidebar"] [data-testid="stCode"] pre,
[data-testid="stSidebar"] pre, [data-testid="stSidebar"] code {{
    background: #0B1530 !important; color: #E7ECF7 !important; }}
[data-testid="stSidebar"] [data-testid="stCode"] * {{ color: #E7ECF7 !important; }}
.side-brand {{ display:flex; gap:.7rem; align-items:center; margin:.2rem 0 1rem 0; }}
.side-brand .logo {{ width:42px; height:42px; border-radius:12px; background:{ACCENT};
    color:{INK}; display:flex; align-items:center; justify-content:center; font-weight:800; }}
.side-brand b {{ font-size:1.02rem; }}
.side-brand span {{ display:block; font-size:.78rem; opacity:.75; }}
.side-title {{ font-weight:700; font-size:1.05rem; margin:.1rem 0 .5rem 0; }}
.side-step {{ font-size:.74rem; font-weight:700; letter-spacing:.06em; color:{ACCENT} !important; }}
.viewing {{ display:flex; gap:.7rem; align-items:center; background:rgba(252,163,17,.12);
    border:1px solid rgba(252,163,17,.5); border-radius:12px; padding:.55rem .7rem; margin-top:.6rem; }}
.viewing .tag {{ background:#fff; color:{INK} !important; border-radius:8px; font-weight:800;
    font-size:.72rem; padding:.45rem .5rem; min-width:2.3rem; text-align:center; }}
.viewing small {{ display:block; opacity:.75; font-size:.72rem; }}

/* hero */
.hero {{ background: linear-gradient(135deg, {INK} 0%, #22407A 100%); border-radius: 28px;
    padding: 2.2rem 2.6rem; color: #fff; margin-bottom: 1.2rem; }}
.hero .row {{ display:flex; gap:.7rem; align-items:center; flex-wrap:wrap; }}
.hero .kicker {{ font-size:.82rem; font-weight:700; color:{ACCENT}; }}
.hero .pill {{ border:1px solid rgba(255,255,255,.35); border-radius:999px; padding:.25rem .8rem;
    font-size:.8rem; font-weight:600; }}
.hero h1 {{ color:#fff; font-size:2.6rem; font-weight:800; margin:.9rem 0 .3rem 0; padding:0; line-height:1.1; }}
.hero .ds {{ color:{ACCENT}; font-weight:700; font-size:1.1rem; margin-bottom:.5rem; }}
.hero p {{ color:#DCE4F5; font-size:1.02rem; margin:0 0 1.1rem 0; }}
.chips {{ display:flex; gap:.6rem; flex-wrap:wrap; }}
.chip {{ border:1px solid rgba(255,255,255,.3); border-radius:10px; padding:.35rem .75rem;
    font-size:.86rem; background:rgba(255,255,255,.06); }}

/* panels */
.panel {{ background:{CARD}; border:1px solid #E3E7F1; border-radius:22px; padding:1.5rem 1.7rem; margin:1rem 0; }}
.panel .eyebrow {{ color:{ACCENT_DARK}; font-weight:700; font-size:.8rem; letter-spacing:.05em; }}
.panel h3 {{ margin:.3rem 0 .3rem 0; padding:0; font-size:1.45rem; font-weight:800; color:{TEXT}; }}
.panel .sub {{ color:{MUTED}; font-size:.94rem; margin-bottom:1rem; }}
.selected {{ border-left:5px solid {ACCENT}; background:#FFF8EA; border-radius:14px; padding:1rem 1.2rem;
    display:flex; gap:1rem; align-items:flex-start; margin-bottom:1.2rem; }}
.selected .tag {{ background:#fff; border:1px solid #F0D9A8; border-radius:10px; font-weight:800;
    font-size:.8rem; padding:.7rem .6rem; min-width:3rem; text-align:center; color:{INK}; }}
.selected small {{ color:{MUTED}; font-weight:600; }}
.selected b.name {{ display:block; font-size:1.2rem; color:{TEXT}; }}
.selected p {{ margin:.3rem 0 0 0; color:{TEXT}; }}
.stats {{ display:grid; grid-template-columns:repeat(3,1fr); background:{CARD}; border:1px solid #E3E7F1;
    border-radius:18px; overflow:hidden; margin:1rem 0; }}
.stats div {{ padding:1rem 1.3rem; border-right:1px solid #E3E7F1; }}
.stats div:last-child {{ border-right:none; }}
.stats small {{ color:{MUTED}; font-weight:700; font-size:.78rem; }}
.stats b {{ display:block; font-size:1.5rem; margin:.15rem 0; color:{TEXT}; }}
.stats span {{ color:{MUTED}; font-size:.84rem; }}
.mgrid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; }}
.mcard {{ border:1px solid #E3E7F1; border-top:4px solid {ACCENT}; border-radius:16px; padding:1rem 1.2rem; background:#fff; }}
.mcard .lab {{ color:{MUTED}; font-weight:600; font-size:.95rem; }}
.mcard .val {{ font-size:2.1rem; font-weight:800; color:{TEXT}; margin:.2rem 0 .7rem 0; }}
.mcard .foot {{ border-top:1px solid #EDF0F6; padding-top:.5rem; display:flex; justify-content:space-between;
    color:{MUTED}; font-size:.83rem; gap:.5rem; }}
.mcard .rng {{ background:{PAPER}; border:1px solid #E3E7F1; border-radius:6px; padding:0 .4rem; font-weight:700; white-space:nowrap; }}
table.t {{ width:100%; border-collapse:collapse; font-size:.93rem; }}
table.t th {{ background:#F0F3FA; color:{MUTED}; text-align:left; padding:.6rem .8rem; border-bottom:1px solid #E3E7F1; }}
table.t td {{ padding:.6rem .8rem; border-bottom:1px solid #EDF0F6; color:{TEXT}; }}
table.t td.n, table.t th.n {{ text-align:right; font-variant-numeric:tabular-nums; }}
table.t td.best {{ background:#FFF1D0; font-weight:800; }}
.winner {{ background:#FFF8EA; border:1px solid #F0D9A8; border-radius:14px; padding:.8rem 1.1rem; margin-top:1rem; color:{TEXT}; }}
.ok {{ background:#E6F4EA; color:#176B34; border-radius:14px; padding:.9rem 1.2rem; margin:1rem 0; font-weight:600; }}
.note {{ color:{MUTED}; font-size:.88rem; margin-top:.7rem; }}
@media (max-width: 900px) {{ .mgrid, .stats {{ grid-template-columns:1fr; }} .hero h1 {{ font-size:1.9rem; }} }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Logic: loading models and computing metrics (no retraining anywhere)
# ----------------------------------------------------------------------------
# --- LOGIC START ---
@st.cache_resource
def load_encoder():
    return joblib.load(BASE_DIR / "model" / "label_encoder.joblib")


@st.cache_resource
def load_model(name):
    return joblib.load(BASE_DIR / MODEL_FILES[name])


def feature_names():
    return list(load_model("Logistic Regression").feature_names_in_)


def compute_metrics(y, pred, proba, n_classes):
    try:
        if n_classes == 2:
            auc = roc_auc_score(y, proba[:, 1])
        else:
            auc = roc_auc_score(y, proba, multi_class="ovr", average="weighted")
    except ValueError:
        auc = float("nan")
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": auc,
        "Precision": precision_score(y, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y, pred, average="weighted", zero_division=0),
        "F1": f1_score(y, pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }


@st.cache_data(show_spinner="Running all five models...")
def run_all_models(X, y):
    enc = load_encoder()
    results = {}
    for name in MODEL_FILES:
        model = load_model(name)
        pred = model.predict(X)
        proba = model.predict_proba(X)
        results[name] = {"metrics": compute_metrics(y, pred, proba, len(enc.classes_)),
                         "pred": pred}
    return results
# --- LOGIC END ---


# ----------------------------------------------------------------------------
# Small HTML helpers
# ----------------------------------------------------------------------------
def html_table(df, number_cols=(), best_cols=(), fmt="{:.4f}"):
    head = "".join(f"<th class='{'n' if c in number_cols else ''}'>{c}</th>" for c in df.columns)
    best_idx = {c: df[c].astype(float).idxmax() for c in best_cols}
    rows = ""
    for idx, row in df.iterrows():
        cells = ""
        for c in df.columns:
            v = row[c]
            cls = "n" if c in number_cols else ""
            if c in best_idx and best_idx[c] == idx:
                cls += " best"
            if c in number_cols and isinstance(v, (float, np.floating)):
                v = fmt.format(v)
            elif isinstance(v, (int, np.integer)):
                v = f"{v:,}"
            cells += f"<td class='{cls.strip()}'>{v}</td>"
        rows += f"<tr>{cells}</tr>"
    return f"<table class='t'><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


def metric_cards(metrics):
    cards = ""
    for k, v in metrics.items():
        desc, rng = METRIC_INFO[k]
        val = "n/a" if np.isnan(v) else f"{v:.4f}"
        cards += (f"<div class='mcard'><div class='lab'>{k}</div><div class='val'>{val}</div>"
                  f"<div class='foot'><span>{desc}</span><span class='rng'>{rng}</span></div></div>")
    return f"<div class='mgrid'>{cards}</div>"


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"""<div class='side-brand'><div class='logo'>ML</div>
        <div><b>{APP_TITLE}</b><span>BITS ID {BITS_ID}</span></div></div>""",
                unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("<div class='side-step'>STEP 1</div><div class='side-title'>Add test data</div>",
                    unsafe_allow_html=True)
        sample_path = BASE_DIR / "test_data.csv"
        if sample_path.exists():
            st.download_button("Download sample test CSV", sample_path.read_bytes(),
                               file_name="test_data.csv", mime="text/csv")
        file = st.file_uploader("Upload test CSV", type="csv",
                                help=f"CSV with all feature columns and the {TARGET} column. Up to {MAX_ROWS:,} rows.")

    with st.container(border=True):
        st.markdown("<div class='side-step'>STEP 2</div><div class='side-title'>Choose a model</div>",
                    unsafe_allow_html=True)
        choice = st.selectbox("Select a model to inspect", list(MODEL_FILES))
        st.markdown(f"<div class='viewing'><div class='tag'>{SHORT[choice]}</div>"
                    f"<div><small>Viewing</small><b>{choice}</b></div></div>",
                    unsafe_allow_html=True)

    with st.expander("Required CSV format"):
        st.write(f"The file needs the column **{TARGET}** (Dropout, Enrolled or Graduate) "
                 "and these 36 feature columns. Column order can differ.")
        st.code("\n".join(feature_names()), language=None)

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown(f"""
<div class='hero'>
  <div class='row'><span class='kicker'>MACHINE LEARNING | CLASSIFICATION</span>
  <span class='pill'>BITS ID {BITS_ID}</span></div>
  <h1>{APP_TITLE}</h1>
  <div class='ds'>Dataset: {DATASET_NAME}</div>
  <p>Upload test data, pick a classifier and see how well it predicts whether a student will drop out, stay enrolled or graduate.</p>
  <div class='chips'><span class='chip'>5 classification models</span><span class='chip'>6 evaluation metrics</span>
  <span class='chip'>36 input features</span><span class='chip'>3 target classes</span></div>
</div>
""", unsafe_allow_html=True)

with st.expander("How the dataset is used"):
    st.write(
        "The full dataset has 4,424 students and 36 features. It was split 80/20 with a fixed random seed "
        "and stratified by outcome. All five models were trained on the 80% part and saved in the `model/` "
        "folder. Only the held-out 20% (885 students) is provided as test data, and the app never retrains "
        "a model on the file you upload.")

# ----------------------------------------------------------------------------
# Empty state
# ----------------------------------------------------------------------------
if file is None:
    st.markdown(f"""<div class='panel'><div class='eyebrow'>GET STARTED</div>
    <h3>Upload a test CSV to see results</h3>
    <div class='sub'>Use the sidebar to download the sample file, then upload it back. Results appear here.</div>
    <table class='t'><tbody>
    <tr><td><b>1.</b> Download or prepare a test CSV with the feature columns and the {TARGET} column</td></tr>
    <tr><td><b>2.</b> Upload it in Step 1 of the sidebar</td></tr>
    <tr><td><b>3.</b> Choose a model in Step 2 and review its metrics, confusion matrix and report</td></tr>
    </tbody></table></div>""", unsafe_allow_html=True)
    st.stop()

# ----------------------------------------------------------------------------
# Read and validate the upload
# ----------------------------------------------------------------------------
try:
    data = pd.read_csv(file)
except Exception as exc:
    st.error(f"The file could not be read as CSV: {exc}")
    st.stop()
data.columns = data.columns.str.strip()

if len(data) == 0:
    st.error("The uploaded file has no rows.")
    st.stop()
if len(data) > MAX_ROWS:
    st.error(f"The file has {len(data):,} rows. Upload at most {MAX_ROWS:,} rows.")
    st.stop()
if TARGET not in data.columns:
    st.error(f"The file must contain the '{TARGET}' column. Open 'Required CSV format' in the sidebar to see all columns.")
    st.stop()

needed = feature_names()
missing = [c for c in needed if c not in data.columns]
if missing:
    st.error(f"{len(missing)} required feature column(s) are missing: " + ", ".join(missing[:8])
             + (" ..." if len(missing) > 8 else ""))
    st.stop()

enc = load_encoder()
labels = data[TARGET].astype(str).str.strip()
unknown = sorted(set(labels) - set(enc.classes_))
if unknown:
    st.error(f"Unknown values in '{TARGET}': {', '.join(unknown)}. Allowed values: {', '.join(enc.classes_)}.")
    st.stop()

X = data[needed]
if X.isnull().any().any():
    st.error("The feature columns contain empty values. Fill or remove them and upload again.")
    st.stop()
y = enc.transform(labels)
classes = [str(c) for c in enc.classes_]

results = run_all_models(X, y)
metrics = results[choice]["metrics"]
pred = results[choice]["pred"]

st.markdown(f"<div class='ok'>{len(data):,} test rows evaluated with {choice}.</div>", unsafe_allow_html=True)
st.markdown(f"""<div class='stats'>
  <div><small>Uploaded test set</small><b>{file.name}</b><span>Current evaluation source</span></div>
  <div><small>Rows being evaluated</small><b>{len(data):,}</b><span>Test data only, no training</span></div>
  <div><small>Input features</small><b>{len(needed)}</b><span>{len(classes)} target classes</span></div>
</div>""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Section 1: metrics
# ----------------------------------------------------------------------------
st.markdown(f"""<div class='panel'><div class='eyebrow'>MODEL RESULTS</div>
<div class='selected'><div class='tag'>{SHORT[choice]}</div><div><small>Selected classifier</small>
<b class='name'>{choice}</b><p>{DEFINITIONS[choice]}</p></div></div>
<h3>{choice}: evaluation metrics</h3>
<div class='sub'>Precision, recall and F1 are weighted averages over the three classes. AUC is one-vs-rest, weighted.</div>
{metric_cards(metrics)}</div>""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Section 2: confusion matrix
# ----------------------------------------------------------------------------
cm = confusion_matrix(y, pred, labels=list(range(len(classes))))
row_sum = cm.sum(axis=1, keepdims=True)
pct = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0) * 100
fig = go.Figure(go.Heatmap(
    z=cm, x=classes, y=classes, xgap=3, ygap=3,
    colorscale=[[0, "#FFF6E0"], [0.5, "#F7B84B"], [1, "#14213D"]],
    colorbar=dict(title="Test records"),
    hovertemplate="Actual %{y}<br>Predicted %{x}<br>%{z} records<extra></extra>"))
top = cm.max() if cm.max() > 0 else 1
for i in range(len(classes)):
    for j in range(len(classes)):
        dark_cell = cm[i][j] / top > 0.55
        fig.add_annotation(
            x=classes[j], y=classes[i], showarrow=False,
            text=f"<b>{cm[i][j]:,}</b><br>{pct[i][j]:.1f}%",
            font=dict(size=16, color="#FFFFFF" if dark_cell else "#14213D"))
fig.update_layout(title=f"{choice}: confusion matrix", xaxis_title="Predicted label",
                  yaxis_title="Actual label", yaxis=dict(autorange="reversed"),
                  height=480, margin=dict(l=20, r=20, t=60, b=20),
                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  font=dict(family="Plus Jakarta Sans, sans-serif", size=14, color=TEXT))
errors = int(cm.sum() - np.trace(cm))

st.markdown("""<div class='panel'><div class='eyebrow'>PREDICTION ERRORS</div>
<h3>Confusion matrix</h3>
<div class='sub'>Rows are the actual class and columns are the predicted class. Percentages are within each actual class.</div>""",
            unsafe_allow_html=True)
st.plotly_chart(fig, width="stretch")
st.markdown(f"<div class='note'>All {len(data):,} test records are shown: {int(np.trace(cm)):,} correct predictions "
            f"and {errors:,} errors.</div></div>", unsafe_allow_html=True)

with st.expander("Confusion matrix values"):
    cm_df = pd.DataFrame(cm, columns=classes)
    cm_df.insert(0, "Actual / Predicted", classes)
    st.markdown(html_table(cm_df, number_cols=classes), unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Section 3: classification report
# ----------------------------------------------------------------------------
rep = classification_report(y, pred, labels=list(range(len(classes))), target_names=classes,
                            output_dict=True, zero_division=0)
rep_df = pd.DataFrame(rep).T.drop(index="accuracy", errors="ignore")
rep_df = rep_df.rename(columns={"f1-score": "f1-score"})
rep_df.insert(0, "Class", rep_df.index)
rep_df["support"] = rep_df["support"].astype(int)
rep_df = rep_df.reset_index(drop=True)

st.markdown(f"""<div class='panel'><div class='eyebrow'>CLASS BY CLASS</div>
<h3>Classification report</h3>
<div class='sub'>Performance for every class in this dataset.</div>
{html_table(rep_df, number_cols=["precision", "recall", "f1-score"])}
<div class='note'>Support is the number of actual records in each class. Undefined precision, recall or F1 is shown as 0.</div>
</div>""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Section 4: all models side by side
# ----------------------------------------------------------------------------
cmp_df = pd.DataFrame([{"ML Model": n, **r["metrics"]} for n, r in results.items()])
best_model = cmp_df.loc[cmp_df["F1"].idxmax(), "ML Model"]
metric_cols = list(METRIC_INFO)

st.markdown(f"""<div class='panel'><div class='eyebrow'>ALL FIVE MODELS</div>
<h3>Model comparison on this test data</h3>
<div class='sub'>Every saved model scored on the file you uploaded. The best value in each column is highlighted.</div>
{html_table(cmp_df, number_cols=metric_cols, best_cols=metric_cols)}
<div class='winner'>Highest F1 on this file: <b>{best_model}</b></div></div>""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Section 5: preview
# ----------------------------------------------------------------------------
with st.expander("Preview test data and predictions"):
    preview = data.head(20).copy()
    preview.insert(0, f"Predicted ({choice})", enc.inverse_transform(pred[:20]))
    st.dataframe(preview, width="stretch")
    st.caption("Showing the first 20 rows. The Target column holds the actual label.")