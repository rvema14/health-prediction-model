import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

def load_data():
    data = pd.read_csv('C:/Users/Rakesh/Downloads/world_health_data.csv')
    data = data.dropna()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.dropna()
    print(data.head())
    return data

def train_model():
    data = load_data()
    X = data[['health_exp', 'maternal_mortality', 'infant_mortality', 'neonatal_mortality', 'under_5_mortality', 'prev_hiv']]
    y = data['life_expect']

    if y.isnull().any() or np.any(np.isinf(y)):
        print("Warning: NaN or infinite values found in the target variable.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    joblib.dump(model, 'xgboost_health_prediction_model.pkl')

def predict_health(data):
    model = joblib.load('xgboost_health_prediction_model.pkl')
    input_data = pd.DataFrame(data, index=[0])
    prediction = model.predict(input_data)
    prediction = float(prediction[0])
    return {'predicted_life_expectancy': prediction[0]}

if __name__ == '__main__':
    train_model()