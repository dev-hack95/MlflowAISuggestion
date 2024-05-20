import os
import sys
import warnings
import numpy as np
import pandas as pd
from logger import logging
from exception import CustomException
from typing import List
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder,  Normalizer

warnings.filterwarnings("ignore")

@dataclass
class DataIngesationConfig:
    train_data_path: str = os.path.join("data/raw/playground-series-s4e4/" + "train.csv")
    test_data_path: str = os.path.join("data/raw/playground-series-s4e4/" + "test.csv")
    train_intermediate_data_path: str = os.path.join("data/interim/" + "train.csv")


class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngesationConfig

    def drop_features(self, df: pd.DataFrame, feature: str) -> pd.DataFrame:
        df = df.drop([feature], axis=1, inplace=True)
        return df

    def initiate_data_ingestion(self, args: str) -> pd.DataFrame:
        try:

            if args == 'train':
                df = pd.read_csv(self.ingestion_config.train_data_path)
            elif args == 'test':
                df = pd.read_csv(self.ingestion_config.test_data_path)

            logging.info(f"{df.head()}")
            logging.info("Successfully loded the data")

            self.drop_features(df, 'id')

            df.to_csv("data/interim/train.csv", index=False)
        except Exception as err:
            raise CustomException(err, sys)
        
    
class Encoding(BaseEstimator, TransformerMixin):
    def __init__(self, column_2: List[str] = ['Sex']):
        self.column_2 = column_2

    def fit(self, df: pd.DataFrame) -> None:
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        oe = OrdinalEncoder()
        df[self.column_2] = oe.fit_transform(df[self.column_2])
        return df
    

class FeatureScaling(BaseEstimator, TransformerMixin):
    def __init__(self, column_1: List[str] = ['Length', 'Diameter', 'Height', 'Whole weight', 'Whole weight.1', 'Whole weight.2', 'Shell weight']):
        self.column_1 = column_1

    def fit(self, df: pd.DataFrame) -> None:
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if (set(self.column_1).issubset(df.columns)):
            norm = Normalizer()
            df[self.column_1] = norm.fit_transform(df[self.column_1])
            return df

class DataTransformation:
    def __init__(self) -> None:
        self.ingestion_config = DataIngesationConfig

    def run_pipeline(self):
        pipe = Pipeline([
            ('encoding', Encoding()),
            ('scaler', FeatureScaling()),
            ])

        transformed_data = pipe.fit_transform(pd.read_csv("data/interim/train.csv"))
        transformed_data = transformed_data.sample(n=20000, random_state=42)
        transformed_data.to_csv("data/processed/train.csv", index=False)
        return transformed_data
        
