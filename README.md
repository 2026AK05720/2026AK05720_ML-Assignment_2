# ML Assignment 2: Predict Students' Dropout and Academic Success

M.Tech (AIML), BITS Pilani WILP | Roll No: 2026AK05720

## 1. Problem Statement
The objective is to predict a student's academic outcome as one of three classes: **Dropout**, **Enrolled** or **Graduate**. Early identification of students at risk of dropping out helps institutions provide timely support and improve retention. Five classification models are trained, compared and demonstrated in a Streamlit web app.

## 2. Dataset Description
- **Source:** UCI Machine Learning Repository, "Predict Students' Dropout and Academic Success"
- **Link:** https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

| Attribute | Value |
|---|---|
| Total records | 4,424 |
| Input features | 36 (demographic, socioeconomic, academic, macroeconomic) |
| Target classes | Dropout (1,421), Enrolled (794), Graduate (2,209) |
| Task type | Multi-class classification (imbalanced) |
| Missing values | None |

**Preprocessing:** column names were stripped of extra whitespace; the target was label-encoded; the data was split 80:20 with stratification (random_state=42); features were standardized with StandardScaler inside each model pipeline. The 20% test portion is saved as `test_data.csv`.

## 3. GitHub Repository
https://github.com/2026AK05720/2026AK05720_ML-Assignment_2

## 4. Models Used

### Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7684 | 0.8961 | 0.7500 | 0.7684 | 0.7531 | 0.6150 |
| Decision Tree | 0.7277 | 0.8339 | 0.7261 | 0.7277 | 0.7242 | 0.5527 |
| KNN | 0.6734 | 0.8154 | 0.6530 | 0.6734 | 0.6576 | 0.4529 |
| Naive Bayes | 0.6588 | 0.8118 | 0.6334 | 0.6588 | 0.6417 | 0.4279 |
| SVM | 0.7582 | 0.8865 | 0.7465 | 0.7582 | 0.7454 | 0.5986 |

Precision, Recall and F1 use weighted averaging, and AUC is one-vs-rest weighted, because the problem has three classes. Recall therefore equals Accuracy.

### Observations

| ML Model | Observation |
|---|---|
| Logistic Regression | Best on every metric (accuracy 0.7684, AUC 0.8961, MCC 0.6150). Strongly predictive features such as first- and second-semester unit results appear to separate the classes well with a mostly linear boundary after scaling. |
| Decision Tree | Accuracy of 0.7277 and MCC of 0.5527, clearly behind Logistic Regression. Its nonlinear splits help, but a single tree is prone to overfitting and has the lower AUC (0.8339) among the top three. |
| KNN | Second-weakest (accuracy 0.6734, MCC 0.4529). With 36 features, distances become less informative, and the model is sensitive to the many weakly relevant features. |
| Naive Bayes | Weakest overall (accuracy 0.6588, MCC 0.4279). It assumes features are independent, but many student attributes (for example semester units enrolled, approved and evaluated) are strongly correlated. |
| Support Vector Machine | Very close second (accuracy 0.7582, AUC 0.8865, MCC 0.5986). The RBF kernel handles scaled features well, but it did not beat the simpler linear model here. |

### Overall Winner
**Logistic Regression** gave the highest Accuracy, AUC, Precision, Recall, F1 and MCC on this dataset, with SVM a close second.

## 5. Streamlit Application
**Live app:** http://localhost:8501/

Features: CSV upload (test data), model selection dropdown, evaluation metrics, confusion matrix and classification report, and a comparison of all models.

## 6. Project structure

```text
ML_Assignment2
│
├── model/
│   ├── Decision_Tree.joblib
│   ├── KNN.joblib
│   ├── label_encoder.joblib
│   ├── Logistic_Regression.joblib
│   ├── Naive_Bayes.joblib
│   └── SVM.joblib
│
├── .gitignore
├── app.py
├── data.csv
├── README.md
├── requirements.txt
├── test_data.csv
└── train_models.py
```

# 7. Run locally
- '1.Install the packages listed in requirements.txt.
- '2.Run python train_models.py once to build the five saved pipelines and the reproducible held-out test CSV.
- '3.Run streamlit run app.py and open the local URL printed in the terminal (normally http://localhost:8501/).
- '4.The dashboard opens with a dataset overview. Download the bundled sample from the sidebar or upload your own labeled test CSV to reveal all five model results, the comparison chart, confusion matrix, report, and metric-led observations.

The app does not train or refit models. All preprocessing and classifier steps are persisted together in each model pipeline. 

 ## 7. Deploy to Streamlit Community Cloud

1. Create a GitHub repository and push the contents of this project folder to its root. Keep `app.py`, `requirements.txt`, `test_data.csv`, and the complete `model/` directory in the repository. Do not upload `venv/`.
2. Open Streamlit Community Cloud and sign in with the GitHub account that can access the repository.
3. Select **Create app**, choose the repository and branch, and set the main file path to `app.py`.
4. Deploy and wait for the build to finish. The app should load using the committed model artifacts; model training is not required in the cloud.
5. Open the resulting public app URL and verify the dataset overview, sample download, CSV upload, five-model comparison, confusion matrix, and classification report.
6. Add both the GitHub repository URL and the deployed app URL to the submission PDF and the links below.

- **GitHub repository:**  https://github.com/2026AK05720/2026AK05720_ML-Assignment_2
- **Live Streamlit app:** https://2026ak05720-mlassignment2.streamlit.app/
## 7. Submission items still to complete
- `Capture the required BITS Virtual Lab screenshot and include it in the final submission PDF.
