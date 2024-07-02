import os
import sys
import warnings
from typing import List
from langchain.chat_models import ChatOllama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import json_loader
from langchain.agents import initialize, Tool, tool
from langchain.graphs import graph_document, networkx_graph
from langchain.prompts import ChatMessagePromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


# Config
warnings.filterwarnings("ignore")

class MLFlowAI:
    def __init__(self) -> None:
        self.system_template = """You are a Data Scientist who have great knowledge of hyperparameter
        your work is to analyze the hyperparameter of machine learning model and suggest the hyperparamter such that
        model accuracy will be enhanced"""

        self.system_template1 = """As a Data Scientist specializing in hyperparameters, 
        your role involves deep analysis and optimization of hyperparameters for machine learning models. 
        your work consist meticulous investigation various hyperparameter configurations to enhance model accuracy and performance"""

        self.model = ChatOllama(
            base_url = "http://localhost:11434/",
            model = "mistral"
        )

    def chat(self, query: List) -> str:
        pass 