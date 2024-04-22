import os
import sys
import yaml
import argparse
import warnings
import mlflow
import pandas as pd
import numpy as np
from typing import Tuple
sys.path.append("./src")
from logger import logging
from dataclasses import dataclass
from catboost import CatBoostRegressor
from utilities.utils import Models
from data.ingestion import DataIngestion, DataTransformation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,  mean_squared_log_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor, ExtraTreesRegressor

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
version = parser.add_argument("--version", type=str, help="Input the version you are running")
args = parser.parse_args()
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(args.version)

@dataclass
class ModelParamsConfig:
    model_params_path: str = os.path.join("parameters" "/params.yml")

class TrainModelV1:
    def __init__(self) -> None:
        self.model_prams_config = ModelParamsConfig
        with open(self.model_prams_config.model_params_path, 'r') as file:
            model_params = yaml.safe_load(file)

        self.models = {
            'lda': LinearDiscriminantAnalysis(),
            'Random_Forest': RandomForestRegressor(n_estimators=model_params["RandomForest"]["params"]["n_estimators"]),
            'Ada_boost': AdaBoostRegressor(n_estimators=model_params["AdaBoost"]["params"]["n_estimators"]),
            'Gradient_boost': GradientBoostingRegressor(n_estimators=model_params["GradientBoost"]["params"]["n_estimators"]),
            'Bagging_Classifer': BaggingRegressor(n_estimators=model_params["Bagging"]["params"]["n_estimators"], max_features=model_params["Bagging"]["params"]["max_features"], max_samples=model_params["Bagging"]["params"]["max_samples"]),
            'Decision_tree': DecisionTreeRegressor(),
            'Extr_tree': ExtraTreesRegressor(n_estimators=model_params["ExtraTree"]["params"]["n_estimators"]),
        }

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        x = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
        return x_train, y_train, x_test, y_test

    def Train(self, model, model_name, x_train, y_train, x_test, y_test):
        with mlflow.start_run(run_name=f"{model_name}_{args.version}"):
            logging.info("* Metrics Logging ")
            logging.info(f"------------------------------ {model_name} ------------------------------------------")
            model = model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            logging.info(f"Mean score: {mean_squared_error(y_pred, y_test)}")
            logging.info(f"Mean SQRT score:  {np.sqrt(mean_squared_error(y_pred, y_test))}")
            logging.info(f"MSLE : { mean_squared_log_error(y_pred, y_test) }")
            logging.info(f"RMSLE : { np.sqrt(mean_squared_log_error(y_pred, y_test)) }")
            logging.info("Metrics Logging end")
            #Models.save_model(path="./models/MachineLearning/", model_name=model_name)
            mlflow.log_metric("MSE", mean_squared_error(y_pred, y_test))
            mlflow.log_metric("RMSE", np.sqrt(mean_squared_error(y_pred, y_test)))
            mlflow.log_metric("MSLE", mean_squared_log_error(y_pred, y_test))
            mlflow.log_metric("RMSLE", np.sqrt(mean_squared_log_error(y_pred, y_test)))
            mlflow.log_params({"model_name": model_name, **model.get_params()})
            mlflow.sklearn.log_model(model, f"{model_name}")
            #mlflow.sklearn.save_model(model, f"./models/MachineLearning/{model_name}")


if __name__ == '__main__':
    DataIngestion().initiate_data_ingestion(args='train')
    data_2 = DataTransformation().run_pipeline()
    train = TrainModelV1()
    x_train, y_train, x_test, y_test = train.split_data(data_2)
    for model_name, model in train.models.items():
        train.Train(model, model_name, x_train, y_train, x_test, y_test)
