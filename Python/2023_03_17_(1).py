import os
from six.moves import urllib
import pandas as pd


def download_data(url, dir_name, file_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    file_path = os.path.join(dir_name, file_name)
    urllib.request.urlretrieve(url, file_path)


dir_name = "datasets/iris"
file_name = "iris.csv"


download_data(
    "https://www.openml.org/data/get_csv/61/dataset_61_iris.arff", dir_name, file_name)

my_data = pd.read_csv("datasets/iris/iris.csv")
my_data.plot(kind="scatter", x='x', y='z')

my_data.info()
