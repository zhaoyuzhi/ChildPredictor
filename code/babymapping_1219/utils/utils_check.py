import sys
import os

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def check_var(tensor, tag=None, exit=True):
    print('[CHECK_VAL]----------> {}'.format(tag))
    print('[CHECK_VAL] Content:\n{}'.format(tensor))
    print('[CHECK_VAL] Type: {}'.format(type(tensor)))
    if isinstance(tensor, list):
        print('[CHECK_VAL] Length: {}'.format(len(tensor)))
    else:
        print('[CHECK_VAL] Shape: {}'.format(tensor.shape))
  #  print('[CHECK_VAL] Max value:{}'.format(max(tensor[0])))
  #  print('[CHECK_VAL] Min value:{}'.format(min(tensor[0])))

    if exit:
        sys.exit()
