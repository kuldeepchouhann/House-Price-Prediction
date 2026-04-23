import pandas as pd, pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df=pd.read_csv('house_data.csv')
X=df[['area','bedrooms','bathrooms','age']]
y=df['price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
pickle.dump(model, open('model.pkl','wb'))
print('Model trained and saved as model.pkl')
