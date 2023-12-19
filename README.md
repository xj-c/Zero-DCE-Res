# Zero-DCE-Res
this is a modify version of Zero-DCE, it use the Residual Block to replace the depthwise separable convolution block. 
# Requirement
1. python3.8
2. pytorch 2.0
3. torchvision
4. opencv
5. cuda 11.8
6. Pillow 9.5.0  
You can also use conda enviroment to run code.  
## Floder structure
```
├── data  
│   ├── test_data # testing data. You can make a new folder for your testing data, like LIME, DICM, and New.  
│   │   ├── DICM   
│   │   └── LIME  
│   │   └── New  
│   └── train_data   
│   │   └── low   
│   │        ├── low.zip # The compressed dataset.   
│   │        └── low.z01  
│   │        └── low.z02  
│   └── result  
│       └── guide.txt  
├── lowlight_test.py # testing code  
├── lowlight_train.py # training code  
├── model.py # Zero-DEC-Res network  
├── dataloader.py  
├── Myloss.py  
├── snapshots  
│   ├── Epoch299.pth #  A pre-trained snapshot (Epoch299.pth)  
```
# Test
Before you run the test, please create the new subfolders in "result" folder which have the same name as the subfolders in "test_data"
```
python lowlight_test.py
```
The script will process the image from the subfolders in "test_data" folder, then write them to the subfolders(same name as subfolders in "test_floder") you created in "result" 
# Train
1. go to the "data/train_data/low" folder
2. unzip the low.zip into the current folder
3. run train script
```
python lowlight_train.py
```
