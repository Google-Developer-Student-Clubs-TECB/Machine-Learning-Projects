import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score,cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score


a=pd.read_csv('D:/ml/ln/archive/Student_Performance.csv')
print(a)
a['Extracurricular Activities']=a['Extracurricular Activities'].replace({"No":0,"Yes":1})
correlation_matrix = a.corr()

# Create a heatmap of the correlation matrix using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()
X=a[['Previous Scores','Sleep Hours','Sample Question Papers Practiced']]
Y=a['Performance Index']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
score=cross_val_score(model,X,Y,cv=10,scoring='r2')
print(score)
msc=np.mean(score)
print(msc)


