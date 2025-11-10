<p align="center">

  <h3 align="center"><a href="https://ieeexplore.ieee.org/document/11091472" target='_blank'>UltraVPR: Unsupervised Lightweight Rotation-Invariant Aerial Visual Place Recognition</a> </h3>
  
</p>

<p align="center">
  Chen, Chao and Li, Chunyu and He, Mengfan and Wang, Jun and Xing, Fei and Meng, Ziyang
</p>

![image](https://github.com/cbbhuxx/UltraVPR/blob/main/img/example0.jpg)

## TODO
- [✅] The ONNX model and its export code will be released.
  The optimized ONNX export code enables the TensorRT model to run at approximately 44 Hz on the Jetson Orin NX, significantly surpassing the 14 Hz reported in the paper.
- [ ] The model weights trained on a larger dataset will be released.

## "Requirements" Note
- mmcv
```
pip install -U openmim
mim install mmcv==1.7.2
```
- faiss
```
conda install -c pytorch faiss-cpu
```



## Dataset
You can download the [training data](https://pan.baidu.com/s/1ips8bresAVJqgzxVBQ86xg?pwd=5kf8) from here.

You can download the test data from here.

[VPAir](https://github.com/AerVisLoc/vpair?tab=readme-ov-file)

[UAV-VisLoc (eval)](https://pan.baidu.com/s/1oF09pLUwQZB5lr1Rq6L0-g?pwd=kau2)

If you need the raw data from UAV-VisLoc:
[UAV-VisLoc](https://github.com/IntelliSensing/UAV-VisLoc)



## Weights

[UltraVPR's weights](https://pan.baidu.com/s/1epVFLysy0WqGh0xUlpTEAA?pwd=1qdf)

## Train
If you want to retrain the model, please first load the pre-trained weights of the backbone network. Relevant code is in utils.py.
[E2ResNet's weights](https://pan.baidu.com/s/13dSCLUrmOH6QhkvlFlJyTg?pwd=ghvf) ([Specific training parameters](https://github.com/cbbhuxx/UltraVPR/blob/main/models/backbones/README.md))

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



















