import matplotlib.pyplot as pyplot
from sklearn import datasets
import numpy as np

digits = datasets.fetch_openml('mnist_784')

X, y = digits.data, digits.target
print(type(X))
print(X.shape)  # DataFrame type

# plot first few images
for i in range(25):
    pyplot.subplot(5, 5, i+1)  # define subplot
    pyplot.imshow(np.array(X.iloc[i]).reshape(
        28, 28), interpolation='nearest', cmap=pyplot.get_cmap('gray'))  # plot raw pixel data

# show the figure
pyplot.show()
