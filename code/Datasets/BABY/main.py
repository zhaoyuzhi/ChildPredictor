import os
import glob
import shutil
import tqdm

def main(sou_dir, dst_dir):
    begin_num=80000
    for root, dirs, files in tqdm.tqdm(os.walk(sou_dir)):
        for item in files:
            if 'png' in item and 'ori' not in item:
                if int(item[:2])>2:   #children
                    source_path = os.path.join(root, item)
                    newpath = os.path.join(dst_dir, str(begin_num)+"_"+item[2:])
                    begin_num += 1
                    shutil.copyfile(source_path, newpath)


if __name__ == '__main__':
    src_dir = './train'
    dst_dir = './only_baby'
    os.makedirs(dst_dir, exist_ok=True)
    main(src_dir, dst_dir)
