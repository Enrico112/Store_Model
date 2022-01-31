import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv('homeprices.csv')

model = linear_model.LinearRegression()
# using df[['area']].values as to ignore header
model.fit(df[['area']].values, df.price)
# using [[x]] to select value as dataframe
predict = model.predict([[5000]])

# APPROACH 1: pickle
import pickle

# write bin file and dump model in it
with open('model_pickle', 'wb') as f:
    pickle.dump(model, f)
# read bin file and load model 
with open('model_pickle', 'rb') as f:
    model = pickle.load(f)
    
predictLoaded = model.predict([[5000]])

# loaded model is equal to original model
predictLoaded == predict

# APPROACH 2: pickle
# joblib is more efficient with numpy arrays
from sklearn.externals import joblib

# write bin file
joblib.dump(model, 'model_joblib')