import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from PIL import Image
from urllib.request import urlopen
import numpy as np

#  import dataset MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#  normalize inputs from 0-255 to 0-1
train_images = train_images / 255
test_images = test_images / 255
#  flatten 28*28 images to a 784 vector for each image
train_images = tf.reshape(tf.cast(train_images, tf.float32), [-1, 28 * 28])
test_images = tf.reshape(tf.cast(test_images, tf.float32), [-1, 28 * 28])
#  one hot vectors outputs
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

#  create the model of NeuralNetwork
model = keras.Sequential([
    layers.Dense(128, activation='relu', name='hidden_layer_1'),
    layers.Dense(10, activation='softmax', name='output')
])
#  compile the model
model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
#  fit the model
model.fit(train_images, train_labels, batch_size=32, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)
#  show the model
model.summary()
#  open image-captcha
url = 'http://localhost/wp-content/uploads/wpcf7_captcha/3065758464.png'
img = Image.open(urlopen(url))
#  resize image
img = img.resize((224, 28), Image.ANTIALIAS)
num = [None] * 8
arr = [None] * 8
result = 0
for i in range(8):
    #  crop captcha to numbers
    num[i] = img.crop((i * 28, 0, (i+1) * 28, 28))
    #  create an array of pixels
    vector = np.array(num[i], dtype='uint8')
    r_chanel = vector[:, :, 0]
    g_chanel = vector[:, :, 1]
    b_chanel = vector[:, :, 2]
    #  greyscale captcha
    vector_grey = r_chanel * 0.2989 + g_chanel * 0.5587 + b_chanel * 0.2989
    vector_grey = vector_grey/255
    #  flatten 28*28 images of numbers to a 784 vector for each image
    vector_grey = tf.reshape(tf.cast(vector_grey, tf.float32), [-1, 28*28])
    #  neural network prediction of each number
    pred = model.predict(vector_grey)
    predict_number = tf.argmax(pred, axis=1).numpy()
    #  captcha solution
    result = result * 10 + predict_number
print(result)