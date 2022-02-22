# Machine Learning Simple Project....
import pandas as pd
import matplotlib.pyplot as mlt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('iphone_price_bd.csv')
# print(data)
# mlt.scatter(data['version'], data['price'])
# mlt.show()
model = LinearRegression()
model.fit(data[['version']], data[['price']])
print(model.predict([[15]]))