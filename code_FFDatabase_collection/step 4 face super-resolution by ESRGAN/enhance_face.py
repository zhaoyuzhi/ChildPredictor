import os
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

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
            continent_name = img_path.split('\\')[-4]
            country_name = img_path.split('\\')[-3]
            count_name = img_path.split('\\')[-2]
            img_name = img_path.split('\\')[-1]
            full_list.append(img_path)
            continent_list.append(continent_name)
            country_list.append(country_name)
            img_list.append(img_name)
    return full_list, continent_list, country_list, count_name, img_list

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

### code for reading model
model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

### code for enhancing face
wholepath = 'C:\\Users\\yzzha\\Desktop\\child faces\\train2'
full_list, continent_list, country_list, count_name, img_list = get_files(wholepath)
savepath = 'C:\\Users\\yzzha\\Desktop\\results'
print('There are overall %d images.' % (len(full_list)))

# judge the image size and enhance the resolution
for i in range(len(full_list)):
    # define path
    read_path = full_list[i]
    save_path = os.path.join(savepath, continent_list[i], country_list[i], count_name[i], img_list[i])
    print('save image as', save_path)
    # check path
    save_folder = os.path.join(savepath, continent_list[i], country_list[i], count_name[i])
    check_path(save_folder)
    # read images
    img = cv2.imread(read_path, cv2.IMREAD_COLOR)
    if img.shape[0] < 64 or img.shape[1] < 64:
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)
        # forward
        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        # resize
        output = cv2.resize(output, (128, 128))
        cv2.imwrite(save_path, output)
    else:
        output = cv2.resize(output, (128, 128))
        cv2.imwrite(save_path, output)
