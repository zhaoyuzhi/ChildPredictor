# ChildPredictor

This is the official webpage of the paper "ChildPredictor: A Child Face Prediction Framework with Disentangled Learning", under Major Revision of IEEE TMM

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

Some files are not included in the current implementation due to privacy issue. After acceptace we will consider to release the pre-trained model for users. The network architectures can be found in the ``code`` folder.

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

Pre-trained models are not released in the current implementation due to privacy issue.

```bash
cd code
cd babymapping_1219
python main.py
```

### Network Architectures

<img src="./img/net.png"/>

<img src="./img/net_x.png"/>

<img src="./img/net_y.png"/>
