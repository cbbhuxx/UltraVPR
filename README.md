# UltraVPR
Unsupervised Lightweight Rotation-Invariant Aerial Visual Place Recognition 

![image](https://github.com/cbbhuxx/UltraVPR/blob/main/img/example0.jpg)

## Prerequisites
- torch
- mmcv
- e2cnn
- faiss
- tensorboardX
- numpy



## Dataset
You can download the [training data](https://cloud.tsinghua.edu.cn/d/68c3a4ed24cc40f1a7da/?p=%2Ftraining_data&mode=list) from here.

You can download the test data from here.

[VPAir](https://github.com/AerVisLoc/vpair?tab=readme-ov-file)

[UAV-VisLoc (eval)](https://drive.google.com/drive/folders/1u7f338ZHbZbMdJPA_a_NPWXX4Y_MgUH9?usp=drive_link)

If you need the raw data from UAV-VisLoc:
[UAV-VisLoc](https://github.com/IntelliSensing/UAV-VisLoc)



## Weights

[UltraVPR's weights](https://drive.google.com/drive/folders/1Vfn6OznzuReX4fcygVY8ASLeTRAQ35b9?usp=drive_link)

[E2ResNet's weights](https://drive.google.com/drive/folders/1-Ft6N4hlR7CDTwNmY0HKYVQnQ54qNc75?usp=drive_link) ([Specific training parameters](https://github.com/cbbhuxx/UltraVPR/blob/main/models/backbone/README.md))


## Train

After preparing the dataset, please modify the path in dataloader/aerialvl.py.
```
python main.py --mode=train
```
If you want to use the model for enhanced training:
```
python main.py --mode=train --upscaling=True
```

## Eval
If you want to test on VPAir, please run the following code:
```
python eval_VPAir.py
```

If you want to test on UAV-VisLoc, please run the following code:
```
python eval_UAV-VisLoc.py
```



