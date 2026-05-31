import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

def run_retraining():
    print("STATUS: Memulai retraining model via MLProject...")
    
    data_dir = 'dataset_preprocessing'
    
    X_train = pd.read_csv(f'{data_dir}/X_train.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv').values.ravel()
    
    with mlflow.start_run() as run:
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
   
        mlflow.sklearn.log_model(rf, "model")
        print(f"STATUS: Retraining sukses. Model tersimpan di run_id: {run.info.run_id}")

if __name__ == "__main__":
    run_retraining()