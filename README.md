# AutoVF  
## AutoVF: Make Automated Vulnerability Fixes Robust with Co-Learning  
Python library dependencies:  
torch -v:1.10.2+cu113  
numpy -v:1.22.3  
tqdm -v:4.62.3  
pandas -v:1.4.1  
datasets -v:2.0.0  
gdown -v:4.5.1  
scikit-learn -v:1.1.2  
tree-sitter -v:0.20.0  
argparse -v:1.4.0  

Dataset:  
Download necessary data and unzip via the following command:  
```bash
cd data
sh download_data.sh 
cd ..
```
## How to reprduce   

  <summary>Environment Setup</summary>
Install the python dependencies via the following command:
  
```bash
cd AutoVF
pip install -r requirements.txt
cd AutoVF/transformers
pip install .
cd ../..
```
We highly recommend you check out this installation guide for the "torch" library so you can install the appropriate version on your device.

To utilize GPU (optional), you also need to install the CUDA library. You may want to check out this installation guide.

Python 3.9.7 is recommended, which has been fully tested without issues.



## If you want to use our model,you need to follow these steps:

## AutoVF (our Approach)
### Retrain Localization Model
```bash
cd AutoVF
sh run_pretrain_loc.sh
sh run_train_loc.sh
cd ..
```
### Retrain Repair Model
```bash
cd AutoVF
sh run_pretrain.sh
sh run_train.sh
sh run_test.sh
cd ..
```

