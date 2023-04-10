from sklearn.datasets import make_regression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X, y = make_regression(n_samples=2400000, n_features=6, noise=0.1, random_state=42)
df = pd.DataFrame(X)
df['target'] = y
df.to_csv('regression_data.csv', index=False)
fig, ax = plt.subplots(nrows=3, ncols=2)
ax[0][0].scatter(X[:, 0], y, s=1)
ax[0][1].scatter(X[:, 1], y, s=2)
ax[1][0].scatter(X[:, 2], y, s=3)
ax[1][1].scatter(X[:, 3], y, s=4)
ax[2][0].scatter(X[:, 4], y, s=5)
ax[2][1].scatter(X[:, 5], y, s=6)
plt.show()
