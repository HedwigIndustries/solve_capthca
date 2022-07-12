import os
import imutils
import cv2

def find_contours(thresh):
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    letter_list = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if (w > (h * 1.2)):
            letter_list.append((x, y, int(w/2), h))
            letter_list.append((x + int(w/2), y, int(w/2), h))
        else:
            letter_list.append((x, y, w, h))
    return letter_list

list_of_counts = {}
def save_letters(letter_list, captcha_solve, grayscale):

    for letter_contour, letter_solve in zip(letter_list, captcha_solve):
        x, y, w, h = letter_contour
        letter = grayscale[y - 2:y + h + 2, x - 2:x + w + 2]
        letter = normalize_shape(letter, 20, 20)
        save_path = os.path.join('extracted_letters', letter_solve)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        count = list_of_counts.get(letter_solve, 1)
        path_to_letter = os.path.join(save_path, "{}.png".format(str(count)))
        cv2.imwrite(path_to_letter, letter)
        list_of_counts[letter_solve] = count + 1

def normalize_shape (image, height, width):
    h = image.shape[0]
    w = image.shape[1]
    if w > h:
        image = imutils.resize(image, width=width)
    else:
        image = imutils.resize(image, height=height)
    board_height = int((height - image.shape[0]) / 2.0)
    board_width = int((width - image.shape[1]) / 2.0)
    image = cv2.copyMakeBorder(image, board_height, board_height,
                               board_width, board_width,
                               cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))
    return image