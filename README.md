# TNet_TMM2022

Runmin Cong, Kepu Zhang, Chen Zhang, Feng Zheng, Yao Zhao, Qingming Huang, and Sam Kwong, Does Thermal really always matter for RGB-T salient object detection?, IEEE Transactions on Multimedia, 2022. In Press.

# Results of  TNet:
* Results:
  - We provide the resutls of our TNet on VT5000, VT1000, VT821 datasets. 
```
Baidu Cloud: https://pan.baidu.com/s/1tdQAnRuUN3F1lJEebUs7fw    Password: xzw4
```

# Pytorch Code of  TNet:
* Pytorch implementation of  TNet
* Pretrained model:
  - We provide our testing code. If you test our model, please download the pretrained model, unzip it, and put the `TNet.pth` to `the_model/` folder.
  - Pretrained model download:
```
Baidu Cloud: https://pan.baidu.com/s/1lrEg-uHPt5Lb2PVEqN4lfQ   Password: iqbe 
```

## Requirements

* Python 3.7
* Pytorch 1.6.0
* torchvision

## Data Preprocessing
* Please download the test data, unzip it, and put the `VT821`, `VT1000`, `VT5000` to `Dataset/` folder.
* train and test datasets:
```
Baidu Cloud: https://pan.baidu.com/s/1mpMKWf-fiN-oqQTepfoDzA   Password: nb9w
```

## Test
```
python test.py
```

* You can find the results in the `'Results/'` folder.

# If you use our TNet, please cite our paper:

    @article{TNet,
     title={Does Thermal Really Always Matter for {RGB-T} Salient Object Detection?},
     author={Cong, Runmin and Zhang, Kepu and Zhang, Chen and Zheng, Feng and Zhao, Yao and Huang, Qingming and Kwong, Sam },
     journal={IEEE Transactions on Multimedia},
     year={early access, doi: 10.1109/TMM.2022.3216476},
    }

# Contact Us:
If you have any questions, please contact Runmin Cong (rmcong@bjtu.edu.cn).
