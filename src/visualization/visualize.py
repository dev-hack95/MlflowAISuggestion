import os
import yaml
import warnings
import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_agraph import Node, agraph, Edge, Config
from sqlalchemy import text
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import sessionmaker

# Config
warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")

st.header("📊 MlFlow Analytics")

class DatabaseSession:
    def __init__(self):
        self.engine = create_engine("sqlite:///playground-series-s4e4.db", echo=True)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def local_session(self):
        return self.SessionLocal()
    
    def query(self, query: str) -> pd.DataFrame:
        return pd.DataFrame(db.execute(text(query)).fetchall())

database = DatabaseSession()
db = database.local_session()
    
class GraphSchema:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.nodes = list()
        self.edges = list()
        self.versions = list()
        self.json_version = list()
        self.config = Config(width=950,
                height=250,
                directed=True, 
                physics=True, 
                hierarchical=False,
                levelSeparation=900,
                )

    def _add_node(self, version, params: dict):
        self.nodes.append(Node(
            id=version,
            title= str(self.model_name) + '_'+ str(version),
            label= str(self.model_name) + '_'+ str(version),
            shape="hexagon",
            size=25,
            color='green',
            info=params,
        ))

    def _add_edge(self, start, end):
        self.edges.append(Edge(
            source=start ,
            target=end,
            ) 
        )

    def _get_data(self, dataframe: pd.DataFrame, metrics: str):
        for idx, row in dataframe.iterrows():
            version = row['experiment_id']
            params = {"key": "value"}
            uid = row['run_uuid']
            params = database.query(f"SELECT key, value from  params where run_uuid = '{uid}' ")
            params_dict = params.set_index('key')['value'].to_dict()
            scores = database.query(f"SELECT key, value from  metrics where run_uuid = '{uid}' AND key = '{metrics}' ")
            scores_dict = scores.set_index('key')['value'].to_dict()
            params = {
                "params": params_dict,
                "scores": scores_dict
            }
            self._add_node(version=version, params=params)

    def _create_edges(self, dataframe: pd.DataFrame):
        for idx, row in dataframe.iterrows():
            current_id = row['experiment_id']
            next_id = current_id + 1
            if next_id in dataframe['experiment_id'].values:
                self._add_edge(current_id, next_id)

    def _add_data(self, dataframe: pd.DataFrame, metrics: str):
        for idx, row, in dataframe.iterrows():
            version = row['experiment_id']
            uid = row['run_uuid']
            params = database.query(f"SELECT key, value from  params where run_uuid = '{uid}' ")
            params_dict = params.set_index('key')['value'].to_dict()
            scores = database.query(f"SELECT key, value from  metrics where run_uuid = '{uid}' AND key = '{metrics}' ")
            scores_dict = scores.set_index('key')['value'].to_dict()
            self.json_version.append({
            version: {
                "params": params_dict,
                "scores": scores_dict
            }})

    def to_json(self):
        return {
            self.model_name: self.json_version
        }


    def _show_graph(self):
        graph = agraph(nodes=self.nodes, edges=self.edges, config=self.config)
        return graph
    
    def _return_data(self):
        return self.nodes


#tables = db.execute(text("SELECT name FROM sqlite_master WHERE type='table';")).fetchall()
#st.write(pd.DataFrame(tables))

tab1, tab2 = st.tabs(['Analytics', "AI Agents"])


with tab1:
    col1, col2 = st.columns([4, 7])

    with col1:
        runs_data = database.query("SELECT DISTINCT(status) as name, COUNT(*) AS value FROM runs GROUP BY status;")
        fig = px.pie(names=runs_data["name"], values=runs_data["value"], hole=0.6, color_discrete_sequence=['#90ee90', '#ff6961'])
        fig.update_traces(pull=[0, 0.05])
        fig.update_traces(textposition='inside', textinfo='value')
        fig.add_annotation(x=0.5,y=0.5,text="Runs",showarrow=False,font=dict(size=20))
        fig.update_layout(showlegend=False, width=400, height=400)
        st.plotly_chart(fig)
    
    with col2:
        experiment_data = database.query("SELECT DISTINCT(name) FROM runs where status != 'FAILED' ")
        selected_option = st.selectbox("Select Experiment", experiment_data['name'])
        run_uuid = db.execute(text(f"select run_uuid from runs where name='{selected_option}'")).fetchone()[0]
        metrics = database.query(f"SELECT * FROM metrics where run_uuid='{run_uuid}'")
        fig = px.bar(metrics, x=metrics['value'], y=metrics['key'])
        fig.update_layout(showlegend=False, width=500, height=300)
        st.plotly_chart(fig)

    col3, col4 = st.columns(2)
    data = database.query("SELECT r.run_uuid, r.experiment_id, m.key, m.value, m.run_uuid as mrun_uuid FROM runs AS r INNER JOIN metrics AS m ON r.run_uuid = m.run_uuid;")
    model_name = database.query("SELECT key as pkey, value as pvalue, run_uuid from params where key='model_name'")
    join_data = pd.merge(data, model_name, on='run_uuid')
    with col3:
        Experiment = st.multiselect(
           "Experiment: ",
           options=join_data['experiment_id'].unique(),
           default=join_data['experiment_id'].unique())
    with col4:
        Metrics = st.selectbox("Select Experiment", join_data['key'].unique())
        
    df_selection = join_data.query( "experiment_id == @Experiment & key == @Metrics")
    fig_1 = px.line(data_frame=df_selection, x=df_selection['pvalue'], y=df_selection['value'], color=df_selection['experiment_id'])
    fig_1.update_layout(showlegend=True, width=1000, height=500)
    st.plotly_chart(fig_1)

with tab2:
     experiment_data_1 = database.query("SELECT name FROM runs")
     experiment_data_1['Algorithm Names'] = [name.split('_v')[0] for name in experiment_data_1['name']]
     metrics_name = database.query("SELECT DISTINCT(key) from metrics")
     col1, col2 = st.columns(2)
     with col1:
         selected_option_1 = st.selectbox("Select Experiment 1", experiment_data_1['Algorithm Names'].unique().tolist())
     with col2:
         selected_metrics = st.selectbox("Select Metrics", metrics_name['key'])
     experiment_data_2 = database.query(f"SELECT * FROM runs WHERE name LIKE '%{selected_option_1}%' and status != 'FAILED';")
     test_1 = GraphSchema(selected_option_1)
     test_1._get_data(experiment_data_2, metrics=selected_metrics)
     test_1._create_edges(experiment_data_2)
     test_1._show_graph()
     #test_1._add_data(experiment_data_2, metrics=selected_metrics)
     output = test_1._return_data()
     #st.write([node.info['params'] for node in output])