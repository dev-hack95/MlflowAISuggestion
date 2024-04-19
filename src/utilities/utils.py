import os
import sys
import pickle
from dataclasses import dataclass
from logger import logging
from exception import CustomException


@dataclass
class ModelPathConfig:
    machine_learning_model_path: str = os.path.join("./models/MachineLearning/")
    deep_learning_model_path: str = os.path.join("./models/DeepLearning/")

class Models:
    def __init__(self) -> None:
        self.model_path_config = ModelPathConfig

    def save_model(self, path: str, model_name: str):
        try:
            with open(path, "wb") as file_obj:
                pickle.dump(model_name, file_obj)
        except Exception as err:
            raise CustomException(err, sys)
        
    def load_model(self, model_path: str):
        try:

            with open(model_path, 'rb') as file:
                model = file.load()

            return model
        
        except Exception as err:
            raise CustomException(err, sys)