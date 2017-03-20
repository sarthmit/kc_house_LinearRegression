import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
from datetime import date
from matplotlib import style
import matplotlib.pyplot as plt

df = pd.read_csv("kc_house_data.csv")
y = np.array(df['price'])
df = df.drop(['id', 'price'],1)
df['year']=df.date.str[0:4]
df['month']=df.date.str[4:6]
df['day']=df.date.str[6:8]
df['year'] = [int(i) for i in df['year']]
df['month'] = [int(i) for i in df['month']]
df['day'] = [int(i) for i in df['day']]
df['days'] = (df['year']-2012)*365 + (df['month']-1)*30 + (df['day']-1)
df = df.drop(['date','year','month','day', 'zipcode'],1)
df['total_size'] = df['sqft_lot'] + df['sqft_living'] + df['sqft_basement']
print(df.head())

df.fillna(value=-99999, inplace=True)
df.dropna(inplace=True)

X = np.array(df)
X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = LinearRegression(n_jobs = -1)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print(accuracy)