import numpy as np
import cv2
import math

def ada_resize(img, size):
    # define image size
    H_in = img.shape[0]
    W_in = img.shape[1]
    # operation
    if (H_in > size) or (W_in > size):
        if H_in <= W_in: # W_in > size definitely
            H_out = size
            W_out = int(math.floor(W_in * float(H_out) / float(H_in)))
        else: # W_in < H_in, and H_in > size definitely
            W_out = size
            H_out = int(math.floor(H_in * float(W_out) / float(W_in)))
    else:
        W_out = W_in
        H_out = H_in
    # resize
    img = cv2.resize(img, (W_out, H_out))
    return img

if __name__ == "__main__":

    # image 1
    img = np.zeros((765, 1829, 3), dtype = np.uint8)
    img = ada_resize(img, 1000)
    print(img.shape)

    # image 2
    img = np.zeros((2765, 1829, 3), dtype = np.uint8)
    img = ada_resize(img, 1000)
    print(img.shape)

    # image 3
    img = np.zeros((575, 828, 3), dtype = np.uint8)
    img = ada_resize(img, 1000)
    print(img.shape)

    # image 4
    img = np.zeros((575, 2828, 3), dtype = np.uint8)
    img = ada_resize(img, 1000)
    print(img.shape)
