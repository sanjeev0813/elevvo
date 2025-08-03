import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("StudentsPerformance.csv")

df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

df['total_score'] = df[['math_score', 'reading_score', 'writing_score']].mean(axis=1)

X = df[['reading_score']]
y = df['total_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_linear = lin_reg.predict(X_test)

print("Linear Regression:")
print("MSE:", mean_squared_error(y_test, y_pred_linear))
print("R2 Score:", r2_score(y_test, y_pred_linear))

plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred_linear, color='red', label='Predicted (Linear)')
plt.xlabel('Reading Score')
plt.ylabel('Total Score')
plt.title('Student Score Prediction - Linear Regression')
plt.legend()
plt.show()

poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_poly_train, y_train)
y_pred_poly = poly_reg.predict(X_poly_test)

print("\nPolynomial Regression (Degree 2):")
print("MSE:", mean_squared_error(y_test, y_pred_poly))
print("R2 Score:", r2_score(y_test, y_pred_poly))

sorted_indices = X_test['reading_score'].argsort()
plt.scatter(X_test, y_test, color='gray', label='Actual')
plt.plot(X_test.iloc[sorted_indices], y_pred_poly[sorted_indices], color='green', label='Predicted (Poly)')
plt.xlabel('Reading Score')
plt.ylabel('Total Score')
plt.title('Student Score Prediction - Polynomial Regression')
plt.legend()
plt.show()