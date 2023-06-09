import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load input data
input_file = 'C:\GIT_KRSuchan\Study_AI\datasets\data_random_forest\data_random_forests.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Split the data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5)

# define the model
model = RandomForestClassifier()
# fit the model
model.fit(X, y)

# get importance
importance = model.feature_importances_

# summarize feature importance
for i, v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i, v))

# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()
