from keras.datasets import mnist
from keras import utils
from keras.models import Sequential
from keras.layers import Dense

# 读取数据，网上预下载的手写数字
path = r'D:\PyProject\Practice-Keras\mnist.npz'
(X_train, y_train), (X_test, y_test) = mnist.load_data(path)

# 二维数组转换成一维数据
X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)

# 转换浮点型
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

# 独热编码
y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)

# 选择顺序模型
model = Sequential()

# 增加输入及第一层
model.add(Dense(512, input_shape=(28*28, ), activation='relu'))

# 增加输出层
model.add(Dense(10, activation='softmax'))

# 编译模型
model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, batch_size=128, epochs=10)

# 模型评价
Testloss, Testaccuracy = model.evaluate(X_test, y_test)

print('Testloss ', Testloss)
print('Testaccuracy: ', Testaccuracy)
