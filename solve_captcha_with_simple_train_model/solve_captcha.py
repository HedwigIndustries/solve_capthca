import cv2
import numpy as np
import pickle
from functions import find_contours
from functions import normalize_shape
from PIL import Image
from urllib.request import urlopen
from keras.models import load_model
import tensorflow as tf

#  import model, dataset, captcha
model = load_model("model.hdf5")
with open("dataset.dat", "rb") as f:
    lb = pickle.load(f)
url = 'http://localhost/wp-content/uploads/wpcf7_captcha/3065758464.png'
captcha = Image.open(urlopen(url))
captcha = np.array(captcha, dtype='uint8')
#  transform captcha to predict
predictions = []
grayscale = cv2.cvtColor(captcha, cv2.COLOR_BGR2GRAY)
grayscale = cv2.copyMakeBorder(grayscale, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=[255, 255, 255])
thresh = cv2.threshold(grayscale, 0, 255,  cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]
letter_list = find_contours(thresh)
#  capthca have four letters
if len(letter_list) != 4:
    print("Error: contours != 4")
#  splitting the captcha into letters and predict letters with train model
letter_list = sorted(letter_list, key=lambda x: x[0])
for letter_contour in letter_list:
    x, y, w, h = letter_contour
    letter = grayscale[y - 2:y + h + 2, x - 2:x + w + 2]
    letter = normalize_shape(letter, 20, 20)
    letter = np.expand_dims(letter, axis=2)
    letter = tf.reshape(tf.cast(letter, tf.float32), [-1, 20 * 20])
    prediction = model.predict(letter)
    letter_predict = lb.inverse_transform(prediction)[0]
    predictions.append(letter_predict)
    captcha_solve = "".join(predictions)

print("CAPTCHA solve is: {}".format(captcha_solve))
