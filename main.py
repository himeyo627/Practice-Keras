from keras.datasets import mnist
import matplotlib.pyplot as plt

path = r'D:\PyProject\Practice-Keras\mnist.npz'

(X_train, y_train), (X_test, y_test) = mnist.load_data(path)
for each in range(4):
   plt.subplot(2,2,each+1)
   plt.imshow(X_train[each], cmap=plt.get_cmap('gray'), interpolation='none')
   plt.title("Class {}".format(y_train[each]))
plt.show()