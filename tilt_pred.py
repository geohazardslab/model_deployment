import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pickle

# print("hello")
df = pd.read_csv('section2new.csv')
df=df.dropna()
print(df)

# df.plot(x="id",y="porepress1")
model= RandomForestRegressor()
X=df[['porepress1','soiltention1','h1x','h1y','h1z','inclination1y']]
X=X[:int(len(df)-1)]
y=df['inclination1x']
y=y[:int(len(df)-1)]
model.fit(X,y)

predictions=model.predict(X)
print('model score:',model.score(X,y))

new_data =df[['porepress1','soiltention1','h1x','h1y','h1z','inclination1y']].tail(1000)
prediction = model.predict(new_data)
print('last row prediction;',prediction)
print('actual value:',df[['inclination1x']].tail(1).values[0][0]) 


pickle.dump(model, open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))
# print(model.predict([[8,91.1,0.025,0.036,0.039,0.22]]))

last_row_prediction = model.predict([[8, 91.1, 0.025, 0.036, 0.039, 0.22]])
print('Last row prediction:', last_row_prediction[0])
