import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/Ragrawal2004/mlops_project_emotion_detection.mlflow")


dagshub.init(repo_owner='Ragrawal2004', repo_name='mlops_project_emotion_detection', mlflow=True)


with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)