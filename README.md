# House-Price-Prediction
 Develop a machine learning model using a housing dataset to predict house prices based  on features

 
from google.colab import files
uploaded = files.upload()

import pandas as pd

# Load dataset
df = pd.read_csv(next(iter(uploaded)))
df.head()
# Step 2: Data Preprocessing
# Display basic info
df.info()

# Handle missing values: fill with median for numerics
df.fillna(df.median(numeric_only=True), inplace=True)

# Get actual column names
print("Columns:", df.columns)

# Define target column and features
target = df.columns[-1]  # Last column is the target
X = df.drop(columns=[target])
y = df[target]

# Identify categorical columns
categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()
print("Categorical features:", categorical_features)

# Apply One-Hot Encoding to categorical features
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# Scale numeric features only
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Now X contains only numeric data

# Done preprocessing
print("Preprocessing complete. Shape:", X_scaled.shape)
# Step 3: Select Features and Target
target = 'price'  # Correct lowercase column name
X = df.drop(columns=[target])
y = df[target]

# Convert categorical features in X using one-hot encoding
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_preds)

# Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)
dt_mse = mean_squared_error(y_test, dt_preds)

print(f"Linear Regression MSE: {lr_mse:.2f}")
print(f"Decision Tree MSE: {dt_mse:.2f}")
# Step 5: Plot predictions
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual Prices', color='blue')
plt.plot(lr_preds, label='Linear Regression Predictions', color='green')
plt.plot(dt_preds, label='Decision Tree Predictions', color='red')
plt.title('House Price Prediction')
plt.xlabel('Sample')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
