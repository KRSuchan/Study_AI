import sklearn.datasets as datasets

iris = datasets.load_iris()
print(type(iris))
print(iris.keys())

targets = iris.feature_names
print(targets)
