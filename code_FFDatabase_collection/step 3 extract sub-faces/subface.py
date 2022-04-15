import math
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import face_recognition
from collections import defaultdict
import numpy as np

import padding
import resizing

def detect_landmark(image_array, model_type = "large"):
    """ return landmarks of a given image array
    :param image_array: numpy array of a single image
    :param model_type: 'large' returns 68 landmarks; 'small' return 5 landmarks
    :return: dict of landmarks for facial parts as keys and tuple of coordinates as values
    """
    face_landmarks_list = face_recognition.face_landmarks(image_array, model = model_type)
    return face_landmarks_list

def align_face(image_array, landmarks):
    """ align faces according to eyes position
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    rotated_img:  numpy array of aligned image
    eye_center: tuple of coordinates for eye center
    angle: degrees of rotation
    """
    # get list landmarks of left and right eye
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    # calculate the mean point of landmarks of left and right eye
    left_eye_center = np.mean(left_eye, axis = 0).astype("int")
    right_eye_center = np.mean(right_eye, axis = 0).astype("int")
    # compute the angle between the eye centroids
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    # compute angle between the line of 2 centeroids and the horizontal line
    angle = math.atan2(dy, dx) * 180. / math.pi
    # calculate the center of 2 eyes
    eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                  (left_eye_center[1] + right_eye_center[1]) // 2)
    # at the eye_center, rotate the image by the angle
    rotate_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale = 1)
    height = int(image_array.shape[0] / 4)
    width = int(image_array.shape[1] / 4)
    image_array = cv2.copyMakeBorder(image_array, height, height, width, width, cv2.BORDER_CONSTANT, value = 0)
    rotated_img = cv2.warpAffine(image_array, rotate_matrix, (image_array.shape[0], image_array.shape[1]))
    return rotated_img, eye_center, angle

def rotate(origin, point, angle, row):
    """ rotate coordinates in image coordinate system
    :param origin: tuple of coordinates,the rotation center
    :param point: tuple of coordinates, points to rotate
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated coordinates of point
    """
    x1, y1 = point
    x2, y2 = origin
    y1 = row - y1
    y2 = row - y2
    angle = math.radians(angle)
    x = x2 + math.cos(angle) * (x1 - x2) - math.sin(angle) * (y1 - y2)
    y = y2 + math.sin(angle) * (x1 - x2) + math.cos(angle) * (y1 - y2)
    y = row - y
    return int(x), int(y)

def rotate_landmarks(landmarks, eye_center, angle, row):
    """ rotate landmarks to fit the aligned face
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :param eye_center: tuple of coordinates for eye center
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated_landmarks with the same structure with landmarks, but different values
    """
    rotated_landmarks = defaultdict(list)
    for facial_feature in landmarks.keys():
        for landmark in landmarks[facial_feature]:
            rotated_landmark = rotate(origin = eye_center, point = landmark, angle = angle, row = row)
            rotated_landmarks[facial_feature].append(rotated_landmark)
    return rotated_landmarks

def corp_face(image_array, landmarks, addsize = 0):
    """ crop face according to eye, mouth and chin position
    :param image_array: numpy array of a single image
    :param size: single int value, size for w and h after crop
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    cropped_img: numpy array of cropped image
    left, top: left and top coordinates of cropping
    """
    x_min1 = np.min(landmarks['chin'], axis = 0)[0]
    x_max1 = np.max(landmarks['chin'], axis = 0)[0]
    x_min2 = np.min(landmarks['left_eyebrow'], axis = 0)[0]
    x_max2 = np.max(landmarks['right_eyebrow'], axis = 0)[0]
    x_min = min(x_min1, x_min2)
    x_max = max(x_max1, x_max2)
    x_center = (x_max - x_min) / 2 + x_min
    
    y_min1 = np.min(landmarks['chin'], axis = 0)[1]
    y_max1 = np.max(landmarks['chin'], axis = 0)[1]
    y_min2 = np.min(landmarks['left_eyebrow'], axis = 0)[1]
    y_max2 = np.max(landmarks['right_eyebrow'], axis = 0)[1]
    y_min = min(y_min1, y_min2)
    y_max = max(y_max1, y_max2)
    y_center = (y_max - y_min) / 2 + y_min

    width = x_max - x_min
    height = y_max - y_min
    length = max(width, height) + addsize
    left = x_center - length / 2
    right = x_center + length / 2
    top = y_center - length / 2
    bottom = y_center + length / 2

    pil_img = Image.fromarray(image_array)
    left, top, right, bottom = [int(i) for i in [left, top, right, bottom]]
    cropped_img = pil_img.crop((left, top, right, bottom))
    cropped_img = np.array(cropped_img)
    return cropped_img, left, top

def transfer_landmark(landmarks, left, top):
    """transfer landmarks to fit the cropped face
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :param left: left coordinates of cropping
    :param top: top coordinates of cropping
    :return: transferred_landmarks with the same structure with landmarks, but different values
    """
    transferred_landmarks = defaultdict(list)
    for facial_feature in landmarks.keys():
        for landmark in landmarks[facial_feature]:
            transferred_landmark = (landmark[0] - left, landmark[1] - top)
            transferred_landmarks[facial_feature].append(transferred_landmark)
    return transferred_landmarks

def face_preprocess(image, landmark_model_type = 'large', addsize = 0):
    """ for a given image, do face alignment and crop face
    :param image: numpy array of a single image
    :param landmark_model_type: 'large' returns 68 landmarks; 'small' return 5 landmarks
    :param crop_size: ingle int value, size for w and h after crop
    :return:
    cropped_face: image array with face aligned and cropped
    transferred_landmarks: landmarks that fit cropped_face
    """
    # detect landmarks
    face_landmarks_dict = detect_landmark(image_array = image, model_type = landmark_model_type)
    cropped_face = []
    transferred_landmarks = []
    for i, face_landmarks_item in enumerate(face_landmarks_dict):
        # rotate image array to align face
        aligned_face, eye_center, angle = align_face(image_array = image, landmarks = face_landmarks_item)
        rotated_landmarks = detect_landmark(image_array = aligned_face, model_type = landmark_model_type)
        sub_cropped_face = []
        sub_transferred_landmarks = []
        for j, face_landmarks_item in enumerate(rotated_landmarks):
            # crop face according to landmarks
            cropped_face_item, left, top = corp_face(image_array = aligned_face, landmarks = rotated_landmarks[j], addsize = addsize)
            # transfer landmarks to fit the cropped face
            transferred_landmarks_item = transfer_landmark(landmarks = rotated_landmarks[j], left = left, top = top)
            sub_cropped_face.append(cropped_face_item)
            sub_transferred_landmarks.append(transferred_landmarks_item)
        # add to the end of list
        cropped_face.append(sub_cropped_face)
        transferred_landmarks.append(sub_transferred_landmarks)
    return cropped_face, transferred_landmarks

def get_image(imgpath, addsize = 0, resize = 1000):
    """ get images
    :param image_array: numpy array of a single image
    :return: the saved image file
    """
    image_array = cv2.imread(imgpath)
    image_array = resizing.ada_resize(image_array, resize)
    print('resizing size:', image_array.shape)
    image_array_pad = padding.out_padding(image_array)
    print('padding size:', image_array_pad.shape)
    # preprocess the face image
    face, landmarks = face_preprocess(image = image_array_pad, landmark_model_type = 'large', addsize = addsize)
    return face, image_array

def save_image(imgpath, addsize = 0, resize = 1000):
    """ save images
    :param image_array: numpy array of a single image
    :return: the saved image file
    """
    image_array = cv2.imread(imgpath)
    
    image_array = resizing.ada_resize(image_array, resize)
    print('resizing size:', image_array.shape)

    image_array_pad = padding.out_padding(image_array)
    print('padding size:', image_array_pad.shape)

    # preprocess the face image
    face, landmarks = face_preprocess(image = image_array_pad, landmark_model_type = 'large', addsize = addsize)
    for i, face_item in enumerate(face):
        for j, sub_face_item in enumerate(face_item):
            new_name = str(i) + '_' + str(j) + ".png"
            cv2.imwrite(new_name, sub_face_item)

if __name__=='__main__':

    # load image
    imgpath = 'C:\\Users\\yzzha\\Desktop\\step 1 result\\train\\Asia\\Afghanistan family\\0.png'
    save_image(imgpath, 0, 1000)
