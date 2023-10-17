import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv('D:/gitt/ml/archive (1)/mcu_box_office.csv')
data = pd.get_dummies(data, columns=['mcu_phase'], prefix='mcu_phase', drop_first=True)
data = data.drop('movie_title', axis=1)
data['release_year'] = pd.to_datetime(data['release_date']).dt.year
data['release_month'] = pd.to_datetime(data['release_date']).dt.month
data['release_day'] = pd.to_datetime(data['release_date']).dt.day
data = data.drop('release_date', axis=1)
data['production_budget'] = data['production_budget'].str.replace(',', '').astype(float)
data['opening_weekend'] = data['opening_weekend'].str.replace(',', '').astype(float)
data['domestic_box_office'] = data['domestic_box_office'].str.replace(',', '').astype(float)
data['worldwide_box_office'] = data['worldwide_box_office'].str.replace(',', '').astype(float)
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()
#movie_title,mcu_phase,release_date,tomato_meter,audience_score,movie_duration,production_budget,opening_weekend,domestic_box_office,worldwide_box_office
X = data[['movie_duration','production_budget','opening_weekend']]
y = data['worldwide_box_office']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)
plt.scatter(y_test, y_pred)
plt.xlabel("True Box Office Revenue")
plt.ylabel("Predicted Box Office Revenue")
plt.title("True vs. Predicted Box Office Revenue")
plt.show()