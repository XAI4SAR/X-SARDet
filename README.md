# X-SARDet

## Introduction
This project is for paper ["Uncertainty Exploration: Toward Explainable SAR Target Detection"](https://ieeexplore.ieee.org/document/10050159)

X-SARDet provides Bayesian deep detectors for SAR ship detection, which provide probabilistic detection results for the users. Here, uncertainty estimation and explanation are conducted for SAR ship detection.

<div align=center>
<img src="https://github.com/XAI4SAR/X-SARDet/blob/main/images/intro.png"/>
</div>


```LaTex
@ARTICLE{huang2023,
  author={Huang, Zhongling and Liu, Ying and Yao, Xiwen and Ren, Jun and Han, Junwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Uncertainty Exploration: Toward Explainable SAR Target Detection}, 
  year={2023},
  volume={61},
  pages={1-14},
  doi={10.1109/TGRS.2023.3247898}}
```

## Features
1. Bayesian deep detectors (BDDs) for horizontal and oriented SAR targets are constructed for uncertainty quantification, answering how much to trust the classification and localization result.

<div align=center>
<img src="https://github.com/XAI4SAR/X-SARDet/blob/main/images/det.png"/>
</div>

2. An occlusion-based explanation method (U-RISE) for BDD is proposed to account for the SAR scattering features that cause uncertainty or promote trustworthiness.

<div align=center>
<img src="https://github.com/XAI4SAR/X-SARDet/blob/main/images/exp.png"/>
</div>

3. Counterfactual analysis is conducted to verify the cause-and-effect relationship between the U-RISE explanation and BDD prediction.

<div align=center>
<img src="https://github.com/XAI4SAR/X-SARDet/blob/main/images/cont.png"/>
</div>

## Getting Started
Code is based on an oriented object detection toolbox, [OBBdetection](https://github.com/jbwang1997/OBBDetection). Please refer to [install.md of OBBdetection](https://github.com/jbwang1997/OBBDetection/blob/master/docs/install.md) for installation and dataset preparation.
