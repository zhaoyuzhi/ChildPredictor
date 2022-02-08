# python3.7
"""Contains the generator class of ProgressiveGAN.

Basically, this class is derived from the `BaseGenerator` class defined in
`base_generator.py`.
"""

import os
import numpy as np

import torch

from Models import model_settings
from Models.pggan_generator_model import PGGANGeneratorModel
from Models.base_generator import BaseGenerator
import torchvision
__all__ = ['PGGANGenerator']


class PGGANGenerator(BaseGenerator):
  """Defines the generator class of ProgressiveGAN."""

  def __init__(self, model_name, logger=None):
    super().__init__(model_name, logger)  #初始化了一堆变量，然后调用build方法
    assert self.gan_type == 'pggan'  

  def build(self):
    self.check_attr('fused_scale')
    self.model = PGGANGeneratorModel(resolution=self.resolution,
                                     fused_scale=self.fused_scale,
                                     output_channels=self.output_channels)

  def load(self):  #加载pytorch版本的模型参数
    self.logger.info(f'Loading pytorch model from `{self.model_path}`.')
    self.model.load_state_dict(torch.load(self.model_path))
    self.logger.info(f'Successfully loaded!')
    self.lod = self.model.lod.to(self.cpu_device).tolist()
    self.logger.info(f'  `lod` of the loaded model is {self.lod}.')

  def convert_tf_model(self, test_num=10):  #第一次将tf版本的参数转化为pytorch版本的参数，同时load给model
    import sys
    import pickle
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    sys.path.append(model_settings.BASE_DIR + '/pggan_tf_official')

    self.logger.info(f'Loading tensorflow model from `{self.tf_model_path}`.')
    tf.InteractiveSession()
    with open(self.tf_model_path, 'rb') as f:
      _, _, tf_model = pickle.load(f)
    self.logger.info(f'Successfully loaded!')

    self.logger.info(f'Converting tensorflow model to pytorch version.')
    tf_vars = dict(tf_model.__getstate__()['variables'])
    state_dict = self.model.state_dict()

    for pth_var_name, tf_var_name in self.model.pth_to_tf_var_mapping.items():
      if 'ToRGB_lod' in tf_var_name:
        lod = int(tf_var_name[len('ToRGB_lod')])
    #    lod_shift = 10 - int(np.log2(self.resolution))
        lod_shift = 0

        tf_var_name = tf_var_name.replace(f'{lod}', f'{lod - lod_shift}')

      if tf_var_name not in tf_vars:
        self.logger.debug(f'Variable `{tf_var_name}` does not exist in '
                          f'tensorflow model.')
        continue
      self.logger.debug(f'  Converting `{tf_var_name}` to `{pth_var_name}`.')
      var = torch.from_numpy(np.array(tf_vars[tf_var_name]))
      if 'weight' in pth_var_name:
        if 'layer0.conv' in pth_var_name:
          var = var.view(var.shape[0], -1, 4, 4).permute(1, 0, 2, 3).flip(2, 3)
        elif 'Conv0_up' in tf_var_name:
          var = var.permute(0, 1, 3, 2)
          #var = var.permute(2, 3, 0, 1)

        else:
          var = var.permute(3, 2, 0, 1)
      state_dict[pth_var_name] = var
    self.logger.info(f'Successfully converted!')

    self.logger.info(f'Saving pytorch model to `{self.model_path}`.')
    torch.save(state_dict, self.model_path)
    self.logger.info(f'Successfully saved!')

    self.load()

    # Official tensorflow model can only run on GPU.
    if test_num <= 0 or not tf.test.is_built_with_cuda():
      return
    self.logger.info(f'Testing conversion results.')
    self.model.eval().to(self.run_device)
    label_dim = tf_model.input_shapes[1][1]
  
    tf_fake_label = np.zeros((1, label_dim), np.float32)
    total_distance = 0.0
    for i in range(test_num):
      latent_code = self.easy_sample(1)
      tf_output = tf_model.run(latent_code, tf_fake_label)
      pth_output = self.synthesize(latent_code)['image']
      distance = np.average(np.abs(tf_output - pth_output))
      self.logger.debug(f'  Test {i:03d}: distance {distance:.6e}.')
      total_distance += distance
    self.logger.info(f'Average distance is {total_distance / test_num:.6e}.')

  def sample(self, num):  #随机采样得到一个sample，用来测试
    assert num > 0
    return np.random.randn(num, self.latent_space_dim).astype(np.float32)

  def preprocess(self, latent_codes): #对采样得到的code进行一个预处理
    if not isinstance(latent_codes, np.ndarray):
      raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')

    latent_codes = latent_codes.reshape(-1, self.latent_space_dim)
    norm = np.linalg.norm(latent_codes, axis=1, keepdims=True)
    latent_codes = latent_codes / norm * np.sqrt(self.latent_space_dim)
    return latent_codes.astype(np.float32)

  def synthesize(self, latent_codes):  #PGGAN的生成方法，传入的是latent code
    if not isinstance(latent_codes, torch.Tensor):
      raise ValueError(f'Latent codes should be with type `torch.Tensor`!')
    latent_codes_shape = latent_codes.shape
    if not (len(latent_codes_shape) == 2 and
            latent_codes_shape[0] <= self.batch_size and
            latent_codes_shape[1] == self.latent_space_dim):
      raise ValueError(f'Latent_codes should be with shape [batch_size, '
                       f'latent_space_dim], where `batch_size` no larger than '
                       f'{self.batch_size}, and `latent_space_dim` equal to '
                       f'{self.latent_space_dim}!\n'
                       f'But {latent_codes_shape} received!')
    zs = latent_codes
  #  zs = torch.from_numpy(latent_codes).type(torch.FloatTensor)
  #  zs = zs.to(self.run_device)
    images = self.model(zs)   #进行inference
    results = {
        'z': latent_codes,   #传入的code
        'image': self.get_value(images),  #对应的结果img，返回的是一个numpy形式的array
    }
    #return results
    return images
