import os
import cv2

# read path
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

# resize all images
def rename_image(read_path, save_path):
    img = cv2.imread(read_path)
    cv2.imwrite(save_path, img)

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    
    # full path for a floder being processed
    wholepath = 'C:\\Users\\yzzha\\Desktop\\child faces\\train2'
    full_list, continent_list, country_list, img_list = get_files(wholepath)
    savepath = 'C:\\Users\\yzzha\\Desktop\\results'
    
    '''
    # calculate the whole number of each class
    once_country_list = [] # only save country name once
    for i, classname in enumerate(country_list):
        if classname not in once_country_list:
            once_country_list.append(classname)
    num_country_list = [] # save the number of images in each country name
    for i, classname in enumerate(once_country_list):
        count = 0
        for j, imgname in enumerate(full_list):
            if imgname.split('\\')[-2] == classname:
                count = count + 1
        num_country_list.append(count)
    '''
    
    # rename all images
    for i in range(len(full_list)):
        # define path
        read_path = full_list[i]
        save_name = str(i) + '.png'
        print('save image as', save_name)
        save_path = os.path.join(savepath, continent_list[i], country_list[i], save_name)
        # check path
        save_folder = os.path.join(savepath, continent_list[i], country_list[i])
        check_path(save_folder)
        # rename
        rename_image(read_path, save_path)
    