# ChildPredictor

This is the official webpage of the paper "ChildPredictor: A Child Face Prediction Framework with Disentangled Learning", accepted to IEEE TMM, 2022

:rocket:  :rocket:  :rocket: **News**:

- **Apr. 15, 2022**: We release the trained models with samples for ChildPredictor.

- **Mar. 31, 2022**: The paper is accepted by the IEEE Transactions on Multimedia.

- **Feb. 8, 2022**: We release the code for ChildPredictor. We are considerring to release the original data of the collected FF-Database.

## 1 FF-Database

We will release the larger-than-ever kinship dataset (FF-Database). Currently, we are asking for legal advice as soon as possible due to the privacy issue.

The data collection pipeline is shown as follows:

<img src="./img/data_collection.png"/>

Some families are shown as follows:

<img src="./img/ffdatabase.png"/>

## 2 Results on Real Families

The generated results on the collected FF-Database:

<img src="./img/sota.png"/>

The generated results on other datasets:

<img src="./img/sota2.png"/>

The disentangled learning analysis is as:

<img src="./img/disentangled_learning_x.png"/>

<img src="./img/disentangled_learning_y.png"/>

The ablation study is as:

<img src="./img/ablation.png"/>

## 3 Implementation

### 3.1 File Structure

Some files are not included in the current implementation since they are too large. The network architectures can be found in the ``code`` folder.

```
code
│
└───baby_model_pool (not provided)
│   └───attgan
│   │   │   attgan_without_claloss_baby.pth
│   │   │   attgan_without_ganloss_celeba_baby.pth
│   │   │   attgan_without_ganloss_claloss_celeba_baby.pth
│   │   │   ...
│   └───inverse
│   │   │   Inverse_ProGAN_GAN_ACGAN_start-with-code.pth
│   │   │   Inverse_ProGAN_GAN_MSGAN_ACGAN_start-with-code.pth
│   │   │   Inverse_ProGAN_GAN_MSGAN_ACGAN_start-with-image.pth
│   │   │   ...
│   └───mapping
│   │   └───Mapping_Xencoder_full_ProGAN_GAN_MSGAN_ACGAN_deepArch_multi-gt_v4
│   │   │   │   MappingNet_Batchsize_32_Epoch_298.pth
│   │   └───Mapping_Xencoder_full_ProGAN_GAN_deepArch_multi-gt_v4
│   │   │   │   MappingNet_Batchsize_32_Epoch_298.pth
│   │   └───Mapping_Xencoder_wo-class_ProGAN_GAN_MSGAN_deepArch_multi-gt_v4
│   │   │   │   MappingNet_Batchsize_32_Epoch_298.pth
│   │   │   ...
│   └───ProGAN-ckp
│   │   │   ProGAN_pt_mixtureData_GAN.pth
│   │   │   ProGAN_pt_mixtureData_GAN_ACGAN.pth
│   │   │   ProGAN_pt_mixtureData_GAN_MSGAN.pth
│   │   │   ProGAN_pt_mixtureData_GAN_MSGAN_ACGAN.pth
│   │   │   ...
│
└───babyinverse (Ey)
│   │   ...
|
└───babymapping_1219 (T)
│   │   ...
│
└───Datasets
│   │   ...
│
└───ProGAN (Gy)
│   │   ...
│
└───AttGAN (please refer to AttGAN official webpage)
│   │   ...
│   
```

### 3.2 Required Libraries

The following packages are needed to be installed:

```bash
pytorch==1.1.0
torchvision==0.3.0
tensorboardx
pyyaml
tqdm
easydict
```

### 3.3 Testing a Real Face

First, download the pre-trained models and unzip them under **code** folder: https://portland-my.sharepoint.com/:f:/g/personal/yzzhao2-c_my_cityu_edu_hk/EoJ0dSnBBgNPnJtCGz108aMBexjNuPU4aF7ePBCzP_yEcQ?e=fkHLuF

Then, you can test some validation samples (we have already put some examples under the **code/babymapping_1219** folder):

```bash
cd code
cd babymapping_1219
python main.py
```

If you want to change the input images, see lines 38-39 of **validation.yaml**: https://github.com/zhaoyuzhi/ChildPredictor/blob/main/code/babymapping_1219/yaml/yaml/validation.yaml

### 3.4 Training

Currently, we do not release the full codes for training due to privacy issue.

### 3.5 Build Your Own Dataset

Please refer to **code_FFDatabase_collection**.

## 4 Network Architectures

<img src="./img/net.png"/>

<img src="./img/net_x.png"/>

<img src="./img/net_y.png"/>

## 5 Reference

```bash
@article{zhao2022childpredictor,
  title={ChildPredictor: A Child Face Prediction Framework with Disentangled Learning},
  author={Zhao, Yuzhi and Po, Lai-Man and Wang, Xuehui and Yan, Qiong and Shen, Wei and Zhang, Yujia and Liu, Wei and Wong Chun-Kit and Pang, Chiu-Sing and Ou, Weifeng and Yu, Wing-Yin and Liu, Buhua},
  journal={IEEE Transactions on Multimedia},
  year={2022}
}
```

## 6 Some Related Works

- Zaman, Ishtiak and Crandall, David. Genetic-GAN: Synthesizing Images Between Two Domains by Genetic Crossover. European Conference on Computer Vision Workshops, 312--326, 2020.

- Gao, Pengyu and Robinson, Joseph and Zhu, Jiaxuan and Xia, Chao and Shao, MIng and Xia, Siyu. DNA-Net: Age and Gender Aware Kin Face Synthesizer. IEEE International Conference on Multimedia and Expo (ICME), 2021.

- Robinson, Joseph Peter and Khan, Zaid and Yin, Yu and Shao, Ming and Fu, Yun. Families in wild multimedia (FIW MM): A multimodal database for recognizing kinship. IEEE Transactions on Multimedia, 2021.
