import os,re,string,warnings,dagshub,mlflow,mlflow.sklearn,matplotlib.pyplot as plt,pandas as pd
from mlflow.models.signature import infer_signature
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,ConfusionMatrixDisplay
warnings.filterwarnings("ignore")

mlflow.set_tracking_uri("https://dagshub.com/Ragrawal2004/mlops_project_emotion_detection.mlflow")
dagshub.init(repo_owner="Ragrawal2004",repo_name="mlops_project_emotion_detection",mlflow=True)
mlflow.set_experiment("LoR Hyperparameter Tuning")

df=pd.read_csv("https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv").drop(columns=["tweet_id"])

lemmatizer=WordNetLemmatizer()
stop_words=set(stopwords.words("english"))

def lower_case(x): return " ".join([i.lower() for i in str(x).split()])
def remove_stop_words(x): return " ".join([i for i in str(x).split() if i not in stop_words])
def removing_numbers(x): return "".join([i for i in str(x) if not i.isdigit()])
def removing_punctuations(x):
    x=re.sub("[%s]"%re.escape(string.punctuation)," ",x)
    return re.sub(r"\s+"," ",x).strip()
def removing_urls(x): return re.sub(r"https?://\S+|www\.\S+","",x)
def lemmatization(x): return " ".join([lemmatizer.lemmatize(i) for i in x.split()])

df["content"]=df["content"].apply(lower_case)
df["content"]=df["content"].apply(remove_stop_words)
df["content"]=df["content"].apply(removing_numbers)
df["content"]=df["content"].apply(removing_punctuations)
df["content"]=df["content"].apply(removing_urls)
df["content"]=df["content"].apply(lemmatization)

df=df[df["sentiment"].isin(["happiness","sadness"])]
df["sentiment"]=df["sentiment"].replace({"sadness":0,"happiness":1})

vectorizer=CountVectorizer()
X=vectorizer.fit_transform(df["content"])
y=df["sentiment"]

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

param_grid={
    "C":[0.1,1,10],
    "penalty":["l1","l2"],
    "solver":["liblinear"]
}

with mlflow.start_run(run_name="GridSearchCV_LogisticRegression") as parent_run:

    mlflow.set_tag("Author","Rounak Agrawal")
    mlflow.set_tag("Project","Twitter Sentiment Detection")
    mlflow.set_tag("Algorithm","Logistic Regression")
    mlflow.set_tag("Vectorizer","Bag of Words")
    mlflow.set_tag("Framework","Scikit-Learn")

    grid_search=GridSearchCV(
        LogisticRegression(),
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    grid_search.fit(X_train,y_train)

    for params,mean_score,std_score in zip(
        grid_search.cv_results_["params"],
        grid_search.cv_results_["mean_test_score"],
        grid_search.cv_results_["std_test_score"]
    ):

        with mlflow.start_run(
            run_name=f"LR_{params}",
            nested=True
        ):

            model=LogisticRegression(**params)

            model.fit(X_train,y_train)

            y_pred=model.predict(X_test)

            accuracy=accuracy_score(y_test,y_pred)
            precision=precision_score(y_test,y_pred)
            recall=recall_score(y_test,y_pred)
            f1=f1_score(y_test,y_pred)

            mlflow.log_params(params)

            mlflow.log_param("Dataset","Twitter Sentiment Dataset")
            mlflow.log_param("Task","Binary Sentiment Classification")
            mlflow.log_param("Train Samples",X_train.shape[0])
            mlflow.log_param("Test Samples",X_test.shape[0])
            mlflow.log_param("Features",X_train.shape[1])

            mlflow.log_metric("Mean CV Score",mean_score)
            mlflow.log_metric("Std CV Score",std_score)
            mlflow.log_metric("Accuracy",accuracy)
            mlflow.log_metric("Precision",precision)
            mlflow.log_metric("Recall",recall)
            mlflow.log_metric("F1 Score",f1)

            print("="*70)
            print(params)
            print(f"Accuracy : {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall   : {recall:.4f}")
            print(f"F1 Score : {f1:.4f}")

            report=classification_report(y_test,y_pred)

            report_file=f"report_{params['C']}_{params['penalty']}.txt"

            with open(report_file,"w") as f:
                f.write(report)

            mlflow.log_artifact(report_file)

            ConfusionMatrixDisplay.from_predictions(
                y_test,
                y_pred,
                cmap="Blues"
            )

            cm_file=f"cm_{params['C']}_{params['penalty']}.png"

            plt.savefig(cm_file,bbox_inches="tight",dpi=300)

            plt.close()

            mlflow.log_artifact(cm_file)

            signature=infer_signature(
                X_train,
                model.predict(X_train)
            )
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=f"LR_C_{params['C']}_{params['penalty']}",
                signature=signature,
                input_example=X_train[:5]
            )

            mlflow.log_artifact(__file__)

            mlflow.set_tag("Author","Rounak Agrawal")
            mlflow.set_tag("Project","Twitter Sentiment Detection")
            mlflow.set_tag("Algorithm","Logistic Regression")
            mlflow.set_tag("Vectorizer","Bag of Words")
            mlflow.set_tag("MLflow Version",mlflow.__version__)

            if os.path.exists(report_file):
                os.remove(report_file)

            if os.path.exists(cm_file):
                os.remove(cm_file)

            print("Run Logged Successfully")
            print("-"*70)

    best_model=grid_search.best_estimator_
    best_params=grid_search.best_params_
    best_score=grid_search.best_score_

    mlflow.log_param("Best C",best_params["C"])
    mlflow.log_param("Best Penalty",best_params["penalty"])
    mlflow.log_param("Best Solver",best_params["solver"])

    mlflow.log_metric("Best CV F1",best_score)

    best_signature=infer_signature(
        X_train,
        best_model.predict(X_train)
    )

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="Best_Logistic_Regression_Model",
        signature=best_signature,
        input_example=X_train[:5]
    )

    mlflow.log_artifact(__file__)

    print("="*80)
    print("GRID SEARCH COMPLETED")
    print("="*80)
    print("Best Parameters")
    print(best_params)
    print(f"Best Cross Validation F1 : {best_score:.4f}")
    print("="*80)