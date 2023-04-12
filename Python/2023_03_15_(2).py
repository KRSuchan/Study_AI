from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

X, y, c = make_regression(n_samples=200, n_features=2,
                          n_informative=2, coef=True)

plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=100)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
