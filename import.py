# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# Dataset
data = {
    'Hours': [1, 2, 3, 4, 5],
    'Marks': [20, 35, 50, 65, 80]
}
# Creating DataFrame
df = pd.DataFrame(data)
# Input and Output
X = df[['Hours']]
y = df['Marks']
# Creating Model
model = LinearRegression()
# Training Model
model.fit(X, y)
# Predicting
predicted = model.predict([[6]])
# Output
print("Predicted Marks for 6 hours study:", predicted[0])
# Plotting Graph
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.show()
