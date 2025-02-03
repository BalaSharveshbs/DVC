import pandas as pd
from sklearn.linear_model import LinearRegression
d=pd.read_csv('data.csv')
x=d[['years']]
y=d['salary']
model = LinearRegression()
model.fit(x,y)
import joblib
joblib.dump(model,'salary.pkl')   