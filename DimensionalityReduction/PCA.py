from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from sklearn import datasets
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

# Load housing data
boston_dataset = datasets.fetch_openml('boston')  # datasets.load_boston()
X = boston_dataset.data

le = LabelEncoder()
X = X.apply(le.fit_transform)
y = boston_dataset.target

# Split data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5)

# 원 데이터를 이용하여 학습 및 성능 측정
classifier = MLPRegressor( hidden_layer_sizes=(5, 2), max_iter=10000, random_state=1)
classifier.fit(X_train, y_train)
predicted = classifier.predict(X_test)
mse = mean_squared_error(y_test, predicted)
print("축소 전 MSE =", round(mse, 2))

pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
k = np.argmax(cumsum >= 0.97) + 1

print("차원 값 :", k)

# k 차원으로 축소
pca = PCA(n_components = k)
X_reduced = pca.fit_transform(X_train)

X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)

# 축소 데이터 이용하여 학습 및 성능 측정
classifier.fit(X_train_reduced, y_train)
y_test_pred_reduced = classifier.predict(X_test_reduced)
mse2 = mean_squared_error(y_test, y_test_pred_reduced)
print("축소 후 MSE =", round(mse2, 2))

