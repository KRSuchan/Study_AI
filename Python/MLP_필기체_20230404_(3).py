import joblib
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

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


digits = load_digits()

# 학습과 테스트 셋 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=30)
# classifier = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=3) # 0.3583333
# classifier = MLPClassifier(hidden_layer_sizes=(
#     5, 2), max_iter=10000, random_state=3) # 0.530555
classifier = MLPClassifier(hidden_layer_sizes=(
    10, 7, 3), max_iter=10000, random_state=3)  # 0.91111
classifier.fit(X_train, y_train)
predicted = classifier.predict(X_test)

print(accuracy_score(y_test, predicted))

joblib.dump(classifier, "digits_cls.pkl", compress=3)
