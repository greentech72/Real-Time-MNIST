import keras
import numpy as np
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from keras.utils import to_categorical

# load data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


def get_noise(p=0.7):
    rand = np.random.rand(28, 28)
    return np.where(rand > p, rand, 0) * 255


# create noise images
x_train_noise = np.array([get_noise() for i in range(10000)])
x_test_noise = np.array([get_noise() for i in range(5000)])

# merge noise & non-noise iamges
x_train = np.concatenate((x_train, x_train_noise), 0)
x_test = np.concatenate((x_test, x_test_noise))
y_train = np.concatenate((y_train, 10*np.ones((10000,))), 0)
y_test = np.concatenate((y_test, 10*np.ones((5000,))), 0)

# shuffle dataset
x_train, y_train = shuffle(x_train, y_train)
x_test, y_test = shuffle(x_test, y_test)

# preprocess dataset
y_train = to_categorical(y_train.astype('uint8'))
y_test = to_categorical(y_test.astype('uint8'))
x_train = x_train / 255.
x_test = x_test / 255.
x_train.shape += (1,)
x_test.shape += (1,)

# define model
model = keras.Sequential(name='MNIST-Model')
model.add(keras.layers.Conv2D(32, 3, padding='same', input_shape=(28, 28, 1), activation='relu', name='conv2d_1'))
model.add(keras.layers.MaxPool2D(2, name='max_pool1'))
model.add(keras.layers.Conv2D(32, 3, padding='same', activation='relu', name='conv2d_2'))
model.add(keras.layers.MaxPool2D(2, name='max_pool2'))
model.add(keras.layers.Flatten())
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(100, activation='relu', name='fc_1'))
model.add(keras.layers.Dense(11, activation='softmax', name='fc_2'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train model (first 1000 images will be used for validation)
history = model.fit(x_train, y_train, validation_data=(x_test[1000:], y_test[1000:]), epochs=5)

# save model
model.save('MNIST_model.h5')