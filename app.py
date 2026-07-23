import pandas as pd
import numpy as np
import shap
import streamlit as slt
from sklearn.ensemble import RandomForestClassifier
import joblib as jb
## import model with the joblib

model = jb.load("dt_model.pkl")

model_columns = jb.load("model_columns.pkl")


