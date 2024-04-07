import numpy as np
import matplotlib.pyplot as plt

# Define values of n
n_values = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]

for n in n_values:
    # Simulate throwing two dice n times
    dice1 = np.random.randint(1, 7, n)
    dice2 = np.random.randint(1, 7, n)
    sum_of_dice = dice1 + dice2

    # Compute frequencies
    h, h2 = np.histogram(sum_of_dice, range(2, 14))

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.bar(h2[:-1], h / n)
    plt.title(f'Histogram of Dice Sum (n = {n})')
    plt.xlabel('Sum of Dice')
    plt.ylabel('Frequency')
    plt.show()

    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    # Load data
    data = pd.read_csv('weight-height.csv')

    # Inspect dependence between height and weight using a scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(data['Height'], data['Weight'], alpha=0.5)
    plt.title('Scatter Plot of Height vs Weight')
    plt.xlabel('Height (inches)')
    plt.ylabel('Weight (pounds)')
    plt.show()

    # Choose appropriate model (linear regression)
    X = data['Height'].values.reshape(-1, 1)
    y = data['Weight'].values

    # Perform regression
    model = LinearRegression()
    model.fit(X, y)

    # Predict weight based on height
    y_pred = model.predict(X)

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.scatter(data['Height'], data['Weight'], alpha=0.5, label='Actual Data')
    plt.plot(data['Height'], y_pred, color='red', label='Linear Regression')
    plt.title('Regression of Weight on Height')
    plt.xlabel('Height (inches)')
    plt.ylabel('Weight (pounds)')
    plt.legend()
    plt.show()

    # Compute RMSE and R2 value
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    print(f'RMSE: {rmse}')
    print(f'R2 Score: {r2}')

    # Assess quality of regression
    # Visual assessment: The regression line appears to capture the general trend of the data.
    # Numerical assessment: RMSE is a measure of the average deviation of predicted values from actual values, and R2 score indicates the proportion of variance explained by the model.




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the data
data = pd.read_csv('weight-height.csv')

# Step 2: Inspect the dependence between height and weight using a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(data['Height'], data['Weight'], alpha=0.5)
plt.title('Scatter Plot of Height vs Weight')
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')
plt.show()

# Step 3: Choose appropriate model for the dependence (Linear Regression)
X = data['Height'].values.reshape(-1, 1)
y = data['Weight'].values

# Step 4: Perform regression on the data using Linear Regression
model = LinearRegression()
model.fit(X, y)

# Step 5: Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(data['Height'], data['Weight'], alpha=0.5, label='Actual Data')
plt.plot(data['Height'], model.predict(X), color='red', label='Linear Regression')
plt.title('Regression of Weight on Height')
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')
plt.legend()
plt.show()

# Step 6: Compute RMSE and R2 value
y_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)
print(f'RMSE: {rmse}')
print(f'R2 Score: {r2}')

# Step 7: Assess the quality of the regression
# Visual assessment: The regression line appears to capture the general trend of the data.
# Numerical assessment: RMSE is a measure of the average deviation of predicted values from actual values, and R2 score indicates the proportion of variance explained by the model.
