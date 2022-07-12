import os
import glob
import cv2
from functions import save_letters
from functions import find_contours

list_of_paths_to_captchas = glob.glob('generated_captchas/*')
list_of_counts = {}
for (i, path_to_captcha) in enumerate(list_of_paths_to_captchas):
    print("[INFO] processing image {}/{}".format(i + 1, len(list_of_paths_to_captchas)))
    captcha_solve = os.path.splitext(os.path.basename(path_to_captcha))[0]
    captcha = cv2.imread(path_to_captcha)
    grayscale = cv2.cvtColor(captcha, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.copyMakeBorder(grayscale, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    thresh = cv2.threshold(grayscale, 0, 255,  cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]
    letter_list = find_contours(thresh)
    #  capthca have four letters
    if len(letter_list) != 4:
        continue

    letter_list = sorted(letter_list, key=lambda x: x[0])
    save_letters(letter_list, captcha_solve, grayscale)
