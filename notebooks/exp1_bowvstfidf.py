
import os
import re
import string
import warnings

import dagshub
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlflow.xgboost
from mlflow.models.signature import infer_signature

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

# ==============================================================
# Configure MLflow + DagsHub
# ==============================================================

mlflow.set_tracking_uri("https://dagshub.com/Ragrawal2004/mlops_project_emotion_detection.mlflow")


dagshub.init(repo_owner='Ragrawal2004', repo_name='mlops_project_emotion_detection', mlflow=True)

# ==============================================================
# Create / Select Experiment
# ==============================================================

mlflow.set_experiment("BoW vs TF-IDF Comparison")

# ==============================================================
# Load Dataset
# ==============================================================

df = pd.read_csv(
    "https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv"
)

# Remove unnecessary column
df = df.drop(columns=["tweet_id"])

print(df.head())

print("\nDataset Shape :", df.shape)

# ==============================================================
# Text Preprocessing Functions
# ==============================================================

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def lower_case(text):
    """Convert text to lowercase."""
    return " ".join([word.lower() for word in str(text).split()])


def remove_stop_words(text):
    """Remove English stop words."""
    return " ".join(
        [
            word
            for word in str(text).split()
            if word not in stop_words
        ]
    )


def removing_numbers(text):
    """Remove digits."""
    return "".join(
        [
            char
            for char in str(text)
            if not char.isdigit()
        ]
    )


def removing_punctuations(text):
    """Remove punctuation."""
    text = re.sub(
        "[%s]" % re.escape(string.punctuation),
        " ",
        text
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def removing_urls(text):
    """Remove URLs."""
    pattern = re.compile(
        r"https?://\S+|www\.\S+"
    )

    return pattern.sub("", text)


def lemmatization(text):
    """Perform lemmatization."""
    text = text.split()

    text = [
        lemmatizer.lemmatize(word)
        for word in text
    ]

    return " ".join(text)


# ==============================================================
# Normalize Complete Dataset
# ==============================================================

def normalize_text(df):

    df["content"] = df["content"].apply(lower_case)

    df["content"] = df["content"].apply(remove_stop_words)

    df["content"] = df["content"].apply(removing_numbers)

    df["content"] = df["content"].apply(removing_punctuations)

    df["content"] = df["content"].apply(removing_urls)

    df["content"] = df["content"].apply(lemmatization)

    return df


df = normalize_text(df)

# ==============================================================
# Keep only Happiness & Sadness
# ==============================================================

mask = df["sentiment"].isin(
    [
        "happiness",
        "sadness"
    ]
)

df = df[mask]

df["sentiment"] = df["sentiment"].replace(
    {
        "sadness": 0,
        "happiness": 1
    }
)

print("\nFinal Dataset Shape :", df.shape)

print(df.head())

# ==============================================================
# Feature Extraction Methods
# ==============================================================

vectorizers = {

    "Bag of Words": CountVectorizer(),

    "TF-IDF": TfidfVectorizer()

}

# ==============================================================
# Machine Learning Algorithms
# ==============================================================

algorithms = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "Multinomial Naive Bayes":
        MultinomialNB(),

    "Random Forest":
        RandomForestClassifier(
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            random_state=42
        ),

    "XGBoost":
        XGBClassifier(
            eval_metric="logloss",
            random_state=42
        )

}


# ==============================================================
# SECTION 2 : PARENT RUN + CHILD RUNS
# ==============================================================

# Parent Run
with mlflow.start_run(run_name="BoW_vs_TFIDF_All_Algorithms") as parent_run:

    # Log parent information
    mlflow.set_tag("Project", "Twitter Sentiment Detection")
    mlflow.set_tag("Author", "Rounak Agrawal")
    mlflow.set_tag("Experiment Type", "Algorithm Comparison")
    mlflow.set_tag("Framework", "Scikit-Learn")
    mlflow.set_tag("Tracking", "MLflow + DagsHub")

    # ==========================================================
    # Iterate through every Algorithm
    # ==========================================================

    for algo_name, algorithm in algorithms.items():

        # ======================================================
        # Iterate through every Vectorizer
        # ======================================================

        for vec_name, vectorizer in vectorizers.items():

            print("=" * 70)
            print(f"Running : {algo_name} + {vec_name}")
            print("=" * 70)

            # ==================================================
            # Child Run
            # ==================================================

            with mlflow.start_run(
                run_name=f"{algo_name}_{vec_name}",
                nested=True
            ):

                # ==============================================
                # Feature Engineering
                # ==============================================

                X = vectorizer.fit_transform(df["content"])

                y = df["sentiment"]

                # ==============================================
                # Train Test Split
                # ==============================================

                X_train, X_test, y_train, y_test = train_test_split(

                    X,
                    y,

                    test_size=0.20,

                    random_state=42,

                    stratify=y

                )

                # ==============================================
                # Log Dataset Information
                # ==============================================

                mlflow.log_param(
                    "Dataset",
                    "Twitter Sentiment Dataset"
                )

                mlflow.log_param(
                    "Task",
                    "Binary Sentiment Classification"
                )

                mlflow.log_param(
                    "Vectorizer",
                    vec_name
                )

                mlflow.log_param(
                    "Algorithm",
                    algo_name
                )

                mlflow.log_param(
                    "Train Samples",
                    X_train.shape[0]
                )

                mlflow.log_param(
                    "Test Samples",
                    X_test.shape[0]
                )

                mlflow.log_param(
                    "Features",
                    X_train.shape[1]
                )

                mlflow.log_param(
                    "Test Size",
                    0.20
                )

                # ==============================================
                # Train Model
                # ==============================================

                model = algorithm

                model.fit(
                    X_train,
                    y_train
                )

                # ==============================================
                # Log Hyperparameters
                # ==============================================

                params = model.get_params()

                for key, value in params.items():

                    try:
                        mlflow.log_param(
                            key,
                            value
                        )

                    except:

                        pass

                # ==============================================
                # Prediction
                # ==============================================

                y_pred = model.predict(
                    X_test
                )

                # ==============================================
                # Calculate Metrics
                # ==============================================

                accuracy = accuracy_score(
                    y_test,
                    y_pred
                )

                precision = precision_score(
                    y_test,
                    y_pred
                )

                recall = recall_score(
                    y_test,
                    y_pred
                )

                f1 = f1_score(
                    y_test,
                    y_pred
                )

                # ==============================================
                # Log Metrics
                # ==============================================

                mlflow.log_metric(
                    "Accuracy",
                    accuracy
                )

                mlflow.log_metric(
                    "Precision",
                    precision
                )

                mlflow.log_metric(
                    "Recall",
                    recall
                )

                mlflow.log_metric(
                    "F1 Score",
                    f1
                )
                                # ==============================================
                # Print Metrics
                # ==============================================

                print(f"Accuracy  : {accuracy:.4f}")
                print(f"Precision : {precision:.4f}")
                print(f"Recall    : {recall:.4f}")
                print(f"F1 Score  : {f1:.4f}")

                # ==============================================
                # Classification Report
                # ==============================================

                report = classification_report(
                    y_test,
                    y_pred
                )

                report_filename = (
                    f"classification_report_"
                    f"{algo_name}_{vec_name}.txt"
                )

                with open(
                    report_filename,
                    "w"
                ) as file:

                    file.write(report)

                mlflow.log_artifact(
                    report_filename
                )

                # ==============================================
                # Confusion Matrix
                # ==============================================

                disp = ConfusionMatrixDisplay.from_predictions(

                    y_test,

                    y_pred,

                    cmap="Blues"

                )

                plt.title(
                    f"{algo_name} - {vec_name}"
                )

                cm_filename = (

                    f"confusion_matrix_"

                    f"{algo_name}_{vec_name}.png"

                )

                plt.savefig(
                    cm_filename,
                    dpi=300,
                    bbox_inches="tight"
                )

                plt.close()

                mlflow.log_artifact(
                    cm_filename
                )

                # ==============================================
                # Infer Model Signature
                # ==============================================

                signature = infer_signature(

                    X_train,

                    model.predict(X_train)

                )

                # ==============================================
                # Log Trained Model
                # ==============================================

                # ==============================================
# Log Trained Model
# ==============================================

# XGBoost uses its own MLflow flavor
                if algo_name == "XGBoost":

        

                 mlflow.xgboost.log_model(

        xgb_model=model,

        artifact_path="model",

        signature=signature,

        input_example=X_train[:5]

                    )

# All other Scikit-Learn models
                else:

                    mlflow.sklearn.log_model(

        sk_model=model,

        artifact_path="model",

        signature=signature,

        input_example=X_train[:5]

                      )

                # ==============================================
                # Log Python Source File
                # ==============================================

                mlflow.log_artifact(
                    __file__
                )

                # ==============================================
                # Log Additional Tags
                # ==============================================

                mlflow.set_tag(
                    "Author",
                    "Rounak Agrawal"
                )

                mlflow.set_tag(
                    "Dataset",
                    "Twitter Emotion Dataset"
                )

                mlflow.set_tag(
                    "Task",
                    "Binary Sentiment Classification"
                )

                mlflow.set_tag(
                    "Vectorizer",
                    vec_name
                )

                mlflow.set_tag(
                    "Algorithm",
                    algo_name
                )

                mlflow.set_tag(
                    "Framework",
                    "Scikit-Learn"
                )

                mlflow.set_tag(
                    "Tracking Server",
                    "DagsHub"
                )

                mlflow.set_tag(
                    "MLflow Version",
                    mlflow.__version__
                )

                print()

                print("-" * 70)

                print(
                    "Run Logged Successfully"
                )

                print("-" * 70)

                print()
                                # =====================================================
                # Cleanup Temporary Files
                # =====================================================

                artifacts = [

                    report_filename,

                    cm_filename

                ]

                for file in artifacts:

                    if os.path.exists(file):

                        os.remove(file)

                # =====================================================
                # Child Run Summary
                # =====================================================

                print("=" * 80)

                print(f"Algorithm   : {algo_name}")

                print(f"Vectorizer  : {vec_name}")

                print(f"Accuracy    : {accuracy:.4f}")

                print(f"Precision   : {precision:.4f}")

                print(f"Recall      : {recall:.4f}")

                print(f"F1 Score    : {f1:.4f}")

                print("=" * 80)

                print()

    # ==========================================================
    # Parent Run Completed
    # ==========================================================

    print()

    print("#" * 90)

    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")

    print("#" * 90)

    print()

    print("Experiments Logged :")

    print()

    print("• Logistic Regression + Bag of Words")

    print("• Logistic Regression + TF-IDF")

    print("• Multinomial Naive Bayes + Bag of Words")

    print("• Multinomial Naive Bayes + TF-IDF")

    print("• Random Forest + Bag of Words")

    print("• Random Forest + TF-IDF")

    print("• Gradient Boosting + Bag of Words")

    print("• Gradient Boosting + TF-IDF")

    print("• XGBoost + Bag of Words")

    print("• XGBoost + TF-IDF")

    print()

    print("Total Child Runs :", len(vectorizers) * len(algorithms))

    print()

    print("Everything has been logged to MLflow successfully.")

    print()

    print("#" * 90)