import os

# ----------------------------------------
#            Create image pair
# ----------------------------------------
def get_basic_folder(path):
    # read a folder, return the family folder name list
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            whole_path = os.path.join(root, filespath)
            delete_len = len(whole_path.split('\\')[-1]) + 1
            whole_path = whole_path[:-delete_len]
            # only save folder name (one such folder may contain many face images)
            if whole_path not in ret:
                ret.append(whole_path)
    return ret

def get_image_pairs(basic_folder_list):
    # read each folder, return each pair; ret is a 2-dimensional list and the smallest dimension represents pair
    ret = []
    for folder_name in basic_folder_list:
        # for a specific family
        for root, dirs, files in os.walk(folder_name):
            # walk this folder
            for filespath in files:
                if filespath != 'ori.png':
                    # parents
                    if int(filespath[:2]) == 1:
                        father = filespath
                    if int(filespath[:2]) == 2:
                        mother = filespath
            # temp saves a training / testing pair
            temp = []
            temp.append(os.path.join(root, father))     # father
            temp.append(os.path.join(root, mother))     # mother
            for filespath in files:
                if filespath != 'ori.png':
                    # children, first two integers > 2
                    if int(filespath[:2]) > 2 and int(filespath[:2]) < 8:
                        temp.append(os.path.join(root, filespath))  # children
            ret.append(temp)
    return ret

path = 'F:\\dataset, my paper related\\Children Face Prediction dataset\\step 4 resized data (128)\\train'
basic_folder_list = get_basic_folder(path)
print(len(basic_folder_list))

ret = get_image_pairs(basic_folder_list)
print(len(ret))
'''
for i in range(10):
    print(ret[i])
'''
print(ret[0][0])
print(len(ret[0]))
