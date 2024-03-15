

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



# Load data
data = pd.read_csv('house_data.csv')



# Print the data
print(data.head())



# Extract features and target variable
X = data[['Square_Footage', 'bedrooms', 'bathrooms']]
y = data['price']


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)



# Function to predict house price based on user input
def predict_house_price(square_footage, bedrooms, bathrooms):
    input_data = np.array([[square_footage, bedrooms, bathrooms]])
    predicted_price = model.predict(input_data)
    return predicted_price[0]



# Prompt user to enter input and give a prediction
print("Enter the following details to predict the house price:")
square_footage = float(input("Square Footage: "))
bedrooms = int(input("Number of Bedrooms: "))
bathrooms = int(input("Number of Bathrooms: "))
predicted_price = predict_house_price(square_footage, bedrooms, bathrooms)
print("Predicted Price:", predicted_price)