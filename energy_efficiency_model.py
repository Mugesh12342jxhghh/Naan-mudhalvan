
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Simulate a dataset (as actual dataset is not provided)
np.random.seed(42)
dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='H')
n_samples = len(dates)

data = pd.DataFrame({
    'datetime': dates,
    'temperature': np.random.normal(25, 5, n_samples),
    'humidity': np.random.uniform(30, 70, n_samples),
    'occupancy': np.random.poisson(10, n_samples),
    'energy_consumption': np.random.normal(200, 50, n_samples) + np.random.uniform(0, 50, n_samples)
})

# Feature engineering
data['hour'] = data['datetime'].dt.hour
data['day_of_week'] = data['datetime'].dt.dayofweek
data = data.drop(columns='datetime')

# Define features and target
X = data.drop(columns='energy_consumption')
y = data['energy_consumption']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Metrics:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test[:500], y=y_pred[:500], alpha=0.6)
plt.xlabel("Actual Energy Consumption")
plt.ylabel("Predicted Energy Consumption")
plt.title("Actual vs Predicted Energy Consumption")
plt.grid(True)
plt.show()
