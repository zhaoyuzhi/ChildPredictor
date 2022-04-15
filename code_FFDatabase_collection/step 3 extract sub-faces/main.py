import argparse
import os
import cv2

import padding
import subface

def get_files(path):
    # read a folder, return the complete path
    full_list = []
    continent_list = []
    country_list = []
    img_list = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            img_path = os.path.join(root, filespath)
            continent_name = img_path.split('\\')[-3]
            country_name = img_path.split('\\')[-2]
            img_name = img_path.split('\\')[-1]
            full_list.append(img_path)
            continent_list.append(continent_name)
            country_list.append(country_name)
            img_list.append(img_name)
    return full_list, continent_list, country_list, img_list

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__=='__main__':

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--wholepath', type = str, default = 'C:\\Users\\yzzha\\Desktop\\step 1 result\\train', help = 'read path')
    parser.add_argument('--savepath', type = str, default = 'C:\\Users\\yzzha\\Desktop\\results', help = 'save path')
    parser.add_argument('--addsize', type = int, default = 0, help = 'additional size for face extraction')
    parser.add_argument('--resize', type = int, default = 1000, help = 'resizing to a fixed size')
    parser.add_argument('--addition_iter', type = int, default = 750, help = 'if it does not begin at 0, addition should be larger than 0')
    opt = parser.parse_args()

    # load image
    full_list, continent_list, country_list, img_list = get_files(opt.wholepath)
    print(len(full_list))

    # loop all the files and get all the faces in one image
    for i in range(len(full_list)):

        i = i + opt.addition_iter

        # get all the faces in one image
        img_path = full_list[i]
        continent_name = continent_list[i]
        country_name = country_list[i]
        img_name = img_list[i]
        print('Now it is the %d-th image with path %s' % (i, img_path))

        # get all the subfaces
        face, ori = subface.get_image(img_path, opt.addsize, opt.resize)
        
        # save the original image
        new_name = 'ori.png'
        new_path = os.path.join(opt.savepath, continent_name, country_name, str(img_name[:-4]), new_name)
        new_folder = os.path.join(opt.savepath, continent_name, country_name, str(img_name[:-4]))
        check_path(new_folder)
        print('Original image will be saved to %s' % (new_path))
        cv2.imwrite(new_path, ori)
        print('original image has been saved!')
        
        # save the subfaces
        for j, face_item in enumerate(face):
            for k, sub_face_item in enumerate(face_item):
                new_name = str(j) + '_' + str(k) + ".png"
                new_path = os.path.join(opt.savepath, continent_name, country_name, str(img_name[:-4]), new_name)
                cv2.imwrite(new_path, sub_face_item)
        print('sub-faces have been saved!')
    