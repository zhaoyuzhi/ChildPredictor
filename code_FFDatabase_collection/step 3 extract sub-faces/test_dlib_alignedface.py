# if the main.py does not work on a single face image
# please try this code
import dlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

# Get the rectangle location of a face
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

# Face alignment
def face_alignment(faces):
    # face landmark detection predictor
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    faces_aligned = []
    for face in faces:
        rec = dlib.rectangle(0,0,face.shape[0],face.shape[1])
        shape = predictor(np.uint8(face), rec)
        # compute the center of two eyes
        eye_center =((shape.part(36).x + shape.part(45).x) * 1./2,
                      (shape.part(36).y + shape.part(45).y) * 1./2)
        # note: right - right
        dx = (shape.part(45).x - shape.part(36).x)
        dy = (shape.part(45).y - shape.part(36).y)
        # compute angle
        angle = math.atan2(dy,dx) * 180. / math.pi
        # compute affine matrix
        RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
        # perform a radiation transformation, ie rotation
        RotImg = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1]))
        faces_aligned.append(RotImg)
    return faces_aligned

# Save the aligned image by one image
def save_by_image(imgpath):
    # pre-process
    im_raw = cv2.imread(imgpath).astype('uint8')
    gray = cv2.cvtColor(im_raw, cv2.COLOR_BGR2GRAY)
    # process
    detector = dlib.get_frontal_face_detector()
    rects = detector(gray, 1)
    src_faces = []
    for (i, rect) in enumerate(rects):
        (x, y, w, h) = rect_to_bb(rect)
        detect_face = im_raw[y : (y + h), x : (x + w)]
        src_faces.append(detect_face)
    # get the aligned face
    faces_aligned = face_alignment(src_faces)
    # save the results
    cv2.imwrite("src.jpg", im_raw)
    i = 0
    for face in faces_aligned:
        cv2.imwrite("det_%d.jpg" % i, face)
        i = i + 1

if __name__ == "__main__":
    
    imgpath = 'C:\\Users\\yzzha\\Desktop\\step 1 result\\train\\Asia\\Arab family\\21.png'
    save_by_image(imgpath)
    