from math import sqrt
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydataset import data
import statistics
import acquire

def prep_iris():
    df = acquire.get_iris_data()
    df = df.drop(columns = ['species_id'])
    df = df.rename(columns={"species_name": "species"})
    df_dummy = pd.get_dummies(df['species'], drop_first=True)
    df= pd.concat([df, df_dummy], axis = 1)
    return df

def prep_telco():
    df = acquire.get_telco_data()
    df = df.drop_duplicates()
    df = df.drop(columns = ['customer_id'])
    df_dummy = pd.get_dummies(df[['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing'  ]], drop_first=[True, True, True, True, True, True, True, True, True, True, True, True])
    df= pd.concat([df, df_dummy], axis = 1)
    return df