import cv2
import numpy as np
import os
from imutils import paths
import pickle
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

#  create dataset
dataset = []
labels = []
for image_file in paths.list_images("extracted_letters"):
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.expand_dims(image, axis=2)
    # otherwise keras won't like it
    label = image_file.split(os.path.sep)[1]
    dataset.append(image)
    labels.append(label)
dataset = np.array(dataset, dtype="float") / 255
labels = np.array(labels)
(train_images, test_images, train_labels, test_labels) = train_test_split(dataset, labels, test_size=0.2, random_state=0)
lb = LabelBinarizer().fit(train_labels)
train_labels = lb.transform(train_labels)
test_labels = lb.transform(test_labels)
train_images = tf.reshape(tf.cast(train_images, tf.float32), [-1, 20 * 20])
test_images = tf.reshape(tf.cast(test_images, tf.float32), [-1, 20 * 20])
#  save dataset
with open("dataset.dat", "wb") as f:
    pickle.dump(lb, f)
#  create the model of NeuralNetwork
model = keras.Sequential([
    #layers.Input(shape= 20),
    layers.Dense(128, activation='relu', input_shape=(400,), name='hidden_layer_1'),
    layers.Dense(32, activation='softmax', name='output')
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
model.save("model.hdf5")