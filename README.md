<p align="center">

  <h3 align="center"><a href="https://ieeexplore.ieee.org/document/11091472" target='_blank'>UltraVPR: Unsupervised Lightweight Rotation-Invariant Aerial Visual Place Recognition</a> </h3>
  
</p>

<p align="center">
  Chen, Chao and Li, Chunyu and He, Mengfan and Wang, Jun and Xing, Fei and Meng, Ziyang
</p>

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

[UAV-VisLoc (eval)](https://pan.baidu.com/s/1oF09pLUwQZB5lr1Rq6L0-g)

If you need the raw data from UAV-VisLoc:
[UAV-VisLoc](https://github.com/IntelliSensing/UAV-VisLoc)



## Weights

[UltraVPR's weights](https://drive.google.com/drive/folders/1Vfn6OznzuReX4fcygVY8ASLeTRAQ35b9?usp=drive_link)

[E2ResNet's weights](https://drive.google.com/drive/folders/1-Ft6N4hlR7CDTwNmY0HKYVQnQ54qNc75?usp=drive_link) ([Specific training parameters](https://github.com/cbbhuxx/UltraVPR/blob/main/models/backbones/README.md))


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

## Acknowledgments 
This work draws inspiration from the following code as references. We extend our gratitude to these remarkable contributions:

- [NetVlad](https://github.com/Nanne/pytorch-NetVlad)
- [ReDet](https://github.com/csuhan/ReDet)
- [CosPlace](https://github.com/gmberton/CosPlace.git)

## License + attribution/citation
When using code within this repository, please refer the following [paper](https://ieeexplore.ieee.org/document/11091472) in your publications:
```
@ARTICLE{11091472,
  author={Chen, Chao and Li, Chunyu and He, Mengfan and Wang, Jun and Xing, Fei and Meng, Ziyang},
  journal={IEEE Robotics and Automation Letters}, 
  title={UltraVPR: Unsupervised Lightweight Rotation- Invariant Aerial Visual Place Recognition}, 
  year={2025},
  volume={10},
  number={9},
  pages={9096-9103}}

```









