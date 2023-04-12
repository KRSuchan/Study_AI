from sklearn.datasets import load_digits
import matplotlib.pyplot as pyplot

digits = load_digits()
print(digits.data.shape)
print(type(digits.data))  # ndarray
x = digits.data.reshape(-1, 8, 8)

# plot first few images
fig = pyplot.figure(figsize=(6, 6))
for i in range(25):
    pyplot.subplot(5, 5, i+1)  # define subplot
    pyplot.imshow(x[i], interpolation='nearest', cmap=pyplot.get_cmap('gray'))

# show the figure
pyplot.show()
