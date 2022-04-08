# ChildPredictor

This is the official webpage of the paper "ChildPredictor: A Child Face Prediction Framework with Disentangled Learning", under IEEE TMM, 2022 (https://ieeexplore.ieee.org/document/9749880)

:rocket:  :rocket:  :rocket: **News**:

- **Mar. 31, 2022**: The paper is accepted by the IEEE Transactions on Multimedia.

- **Feb. 8, 2022**: We are considerring to release the original data of the collected FF-Database.

- **Feb. 8, 2022**: We release the code for ChildPredictor.

## FF-Database

We will release the larger-than-ever kinship dataset (FF-Database) after the publication.

The data collection pipeline is shown as follows:

<img src="./img/data_collection.png"/>

Some families are shown as follows:

<img src="./img/ffdatabase.png"/>

## Results on Real Families

The generated results on the collected FF-Database:

<img src="./img/sota.png"/>

The generated results on other datasets:

<img src="./img/sota2.png"/>

The disentangled learning analysis is as:

<img src="./img/disentangled_learning_x.png"/>

<img src="./img/disentangled_learning_y.png"/>

The ablation study is as:

<img src="./img/ablation.png"/>

## Implementation

### File Structure

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

### Testing a Real Face

Pre-trained models are not released in the current implementation due to privacy issue. We will ask for legal advice as soon as possible.

```bash
cd code
cd babymapping_1219
python main.py
```

### Network Architectures

<img src="./img/net.png"/>

<img src="./img/net_x.png"/>

<img src="./img/net_y.png"/>

## Some Related Works

- Zaman, Ishtiak and Crandall, David. Genetic-GAN: Synthesizing Images Between Two Domains by Genetic Crossover. European Conference on Computer Vision Workshops, 312--326, 2020.

- Gao, Pengyu and Robinson, Joseph and Zhu, Jiaxuan and Xia, Chao and Shao, MIng and Xia, Siyu. DNA-Net: Age and Gender Aware Kin Face Synthesizer. IEEE International Conference on Multimedia and Expo (ICME), 2021.

- Robinson, Joseph Peter and Khan, Zaid and Yin, Yu and Shao, Ming and Fu, Yun. Families in wild multimedia (FIW MM): A multimodal database for recognizing kinship. IEEE Transactions on Multimedia, 2021.

## Reference

```bash
@article{zhao2022childpredictor,
  title={ChildPredictor: A Child Face Prediction Framework with Disentangled Learning},
  author={Zhao, Yuzhi and Po, Lai-Man and Wang, Xuehui and Yan, Qiong and Shen, Wei and Zhang, Yujia and Liu, Wei and Wong Chun-Kit and Pang, Chiu-Sing and Ou, Weifeng and Yu, Wing-Yin and Liu, Buhua},
  journal={IEEE Transactions on Multimedia},
  year={2022}
}
```
