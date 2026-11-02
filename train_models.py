import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

TARGET = "Target"
df = pd.read_csv("data.csv", sep=";")
df.columns = df.columns.str.strip()

enc = LabelEncoder()
y = enc.fit_transform(df[TARGET])
X = df.drop(columns=[TARGET])

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(probability=True, random_state=42),
}

os.makedirs("model", exist_ok=True)
rows = []
for name, clf in models.items():
    pipe = Pipeline([("scale", StandardScaler()), ("clf", clf)])
    pipe.fit(X_tr, y_tr)
    joblib.dump(pipe, f"model/{name.replace(' ', '_')}.joblib")

    pred = pipe.predict(X_te)
    proba = pipe.predict_proba(X_te)
    if len(enc.classes_) == 2:
        auc = roc_auc_score(y_te, proba[:, 1])
    else:
        auc = roc_auc_score(y_te, proba, multi_class="ovr", average="weighted")
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_te, pred),
        "AUC": auc,
        "Precision": precision_score(y_te, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_te, pred, average="weighted"),
        "F1": f1_score(y_te, pred, average="weighted"),
        "MCC": matthews_corrcoef(y_te, pred),
    })

joblib.dump(enc, "model/label_encoder.joblib")

test_df = X_te.copy()
test_df[TARGET] = enc.inverse_transform(y_te)
test_df.to_csv("test_data.csv", index=False)

print(pd.DataFrame(rows).round(4).to_string(index=False))