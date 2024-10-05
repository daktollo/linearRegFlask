import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib


# Load the dataset
data = pd.read_csv('Height-Weight_Dataset.csv')



# Use 'Boy (cm)' as the feature and 'Kilo (kg)' as the target
X = data[['Boy (cm)']]
y = data['Kilo (kg)']


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")

# Save the model to a file
joblib.dump(model, 'linear_regression_model.pkl')

print("Model training complete. Model saved as 'linear_regression_model.pkl'.")