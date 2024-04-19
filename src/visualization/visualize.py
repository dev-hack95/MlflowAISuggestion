import os
import yaml
import warnings
import pandas as pd
import streamlit as st
import streamlit_shadcn_ui as ui
import plotly.express as px
from sqlalchemy import text, inspect
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# Config
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
        experiment_data = database.query("SELECT name FROM runs")
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