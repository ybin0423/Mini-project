import pandas as pd
import numpy as np
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

def evaluate_model(predictions, actual):
    mse = mean_squared_error(actual, predictions)
    r2 = r2_score(actual, predictions)
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)
   
df = pd.read_csv('C:\\Users\\user\\Documents\\466\\Mini Project\\BTC-USD.csv')
df.isnull().sum()
df['Date'] = pd.to_datetime(df['Date'])
df.dropna(inplace=True)

required_features = ['Open', 'High', 'Low','Volume']
output_label = 'Close'
x_lab = ['Date']

# Split the dataset into train, validation, and test sets
train_size = 0.6
val_size = 0.2
test_size = 0.2

x_train_temp, x_remaining, y_train_temp, y_remaining = train_test_split(
    df[required_features], df[output_label], test_size=(val_size + test_size), random_state=30)

x_val, x_test, y_val, y_test = train_test_split(
    x_remaining, y_remaining, test_size=0.5, shuffle=True, random_state=30)

x_train_val = pd.concat([x_train_temp, x_val], axis=0)
y_train_val = pd.concat([y_train_temp, y_val], axis=0)

_, d_set, _, _ = train_test_split(
    df[x_lab], df[output_label], test_size=(val_size + test_size), shuffle=True, random_state=30)

#Decision Tree
dt_model = DecisionTreeRegressor()
param_grid_dt = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2, 4]
}
dt_grid = GridSearchCV(dt_model, param_grid_dt, scoring='neg_mean_squared_error', cv=5)
dt_grid.fit(x_train_val, y_train_val)
best_params_dt = dt_grid.best_params_

dt_model = DecisionTreeRegressor(**best_params_dt)
dt_model.fit(x_train_val, y_train_val)

# Ridge Regression
ridge_model = Ridge()
param_grid = {'alpha': [0,1, 0.001, 1]}
ridge_grid = GridSearchCV(ridge_model, param_grid, scoring='neg_mean_squared_error', cv=5)
ridge_grid.fit(x_train_val, y_train_val)
best_alpha = ridge_grid.best_params_

ridge_model = Ridge(**best_alpha)
ridge_model.fit(x_train_val, y_train_val)

# XGBoost Regression
xgb_model = XGBRegressor()
param_grid = {
    'learning_rate': [0.01, 0.1],
    'n_estimators': [30, 50],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

xgb_grid = GridSearchCV(xgb_model, param_grid, scoring='neg_mean_squared_error', cv=5)
xgb_grid.fit(x_train_val, y_train_val)
best_params_xgb = xgb_grid.best_params_

xgb_model = XGBRegressor(**best_params_xgb)
xgb_model.fit(x_train_val, y_train_val)
future_set = x_test[-21:][required_features]  
future_set_dates = df[-21:]['Date']  

print("Decision Tree:")
print("Best Parameters:", dt_grid.best_params_)
dt_prediction = dt_model.predict(future_set)
evaluate_model(dt_prediction, y_test[-21:])

print("\nRidge Regression:")
print("Best Alpha:", best_alpha)
ridge_future_prediction = ridge_model.predict(future_set)
evaluate_model(ridge_future_prediction, y_test[-21:])

print("\nXGBoost Regression:")
print("Best Parameters:", xgb_grid.best_params_)
xgb_future_prediction = xgb_model.predict(future_set)
evaluate_model(xgb_future_prediction, y_test[-21:])

plt.subplot(3,1,1)
plt.plot(future_set_dates, y_test[-21:], color='blue', lw=2, label='Closed Price')
plt.plot(future_set_dates, ridge_future_prediction, color='r', lw=2, label='Ridge Regression Future Prediction')
plt.title("Ridge Regression of Bitcoin Price Prediction within 3 weeks")
plt.xlabel("Date")
plt.ylabel("Bitcoin Price")
plt.legend()

plt.subplot(3,1,2)
plt.plot(future_set_dates, y_test[-21:], color='blue', lw=2, label='Closed Price')
plt.plot(future_set_dates, dt_prediction, color='red', lw=2, label='Decision Tree Validation Prediction')
plt.title("Decision Tree Regression of Bitcoin Price Prediction within 3 weeks")
plt.xlabel("Date")
plt.ylabel("Bitcoin Price")
plt.legend()

plt.subplot(3,1,3)
plt.plot(future_set_dates, y_test[-21:], color='blue', lw=2, label='Closed Price')
plt.plot(future_set_dates, xgb_future_prediction, color='r', lw=2, label='XGBoost Future Prediction')
plt.title("XGBoost Regression of Bitcoin Price Prediction within 3 weeks")
plt.xlabel("Date")
plt.ylabel("Bitcoin Price")
plt.legend()
plt.tight_layout()
plt.show()







