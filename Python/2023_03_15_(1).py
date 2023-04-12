import os
import numpy as np


def create_learning_data(file_name):
    f = open(file_name, "w")
    f.write("x.y.z\n")

    np.random.seed(30)
    x = 50*np.random.rand(100)
    y = 50*np.random.rand(100)
    z = 3*x+2*y+4+np.random.randn(100)  # np.random.randn(100) 노이즈를 일으키는 역할

    for v1, v2, v3 in zip(x, y, z):  # zip(x,y,z): x,y,z를 하나로 묶어줌
        f.write(str(v1) + ","+str(v2)+"," + str(v3)+"\n")

    f.close()


create_learning_data("sample1.txt")
