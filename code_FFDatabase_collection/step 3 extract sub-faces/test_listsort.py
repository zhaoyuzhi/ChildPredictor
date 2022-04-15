import os

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

if __name__=='__main__':

    # load image
    wholepath = 'C:\\Users\\yzzha\\Desktop\\step 1 result\\train'
    full_list, continent_list, country_list, img_list = get_files(wholepath)
    print(full_list[:30])
    full_list.sort()
    print(full_list[:30])
    #print(continent_list)
    print(img_list[0][:-4])
