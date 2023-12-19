# Zero-DEC-Res
this is a modify version of Zero-DCE, it use the Residual Block to replace the depthwise separable convolution block. 
# Requirement
1. python3.8
2. pytorch 2.0
3. torchvision
4. opencv
5. cuda 11.8
You can also use conda enviroment to run code.
## Floder structure
├── data
│   ├── test_data # testing data. You can make a new folder for your testing data, like LIME, DICM, and New.
│   │   ├── DICM 
│   │   └── LIME
│   │   └── New
│   └── train_data 
│   │   ├── low 
│   │        ├── low.zip # The compressed dataset. 
│   │        └── low.z01
│   │        └──  low.z02
│   └── result
│         └── guide.txt
├── lowlight_test.py # testing code
├── lowlight_train.py # training code
├── model.py # Zero-DEC-Res network
├── dataloader.py
├── Myloss.py
├── snapshots
│   ├── Epoch299.pth #  A pre-trained snapshot (Epoch299.pth)
