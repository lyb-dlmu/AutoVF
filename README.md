# AutoVF  
## AutoVF:Make Automated Vulnerability Fixes Robust with Co-Learning  
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
# Table of contents  
## How to reprduce   
<details>
  <summary>Environment Setup</summary>

First of all, clone this repository to your local machine and access the main directory via the following command:

```bash
```
git clone https://github.com/awsm-research/VQM.git
cd AutoVF
```
Then, install the python dependencies via the following command:
```
pip install -r requirements.txt
cd AutoVF/transformers
pip install .
cd ../..
```
We highly recommend you check out this installation guide for the "torch" library so you can install the appropriate version on your device.

To utilize GPU (optional), you also need to install the CUDA library. You may want to check out this installation guide.

Python 3.9.7 is recommended, which has been fully tested without issues.

</details> ```


