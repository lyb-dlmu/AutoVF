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
First of all, clone this repository to your local machine and access the main dir via the following command:
```bash
git clone https://github.com/awsm-research/VQM.git
cd AutoVF
```
Then, install the python dependencies via the following command:
  
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




### Reproduction of Experiments
#### Reproduce Section 4 - RQ1

- **AutoVF(Proposed Approach)**

    - **Retrain Localization Model**
        ```bash
        cd AutoVF
        sh run_pretrain_loc.sh
        sh run_train_loc.sh
        cd ..
        ```

    - **Retrain Repair Model**
       ```bash
       cd AutoVF
       sh run_pretrain.sh
       sh run_train.sh
       sh run_test.sh
       cd ..
       ```
 - **VQM**

    - **Inference**
        ```bash
        cd VQM/saved_models/checkpoint-best-loss
        sh download_models.sh
        cd ../..
        sh run_test.sh
        cd ..
        ```

    - **Retrain Localization Model**
       ```bash
       cd VQM
       sh run_pretrain_loc.sh
       sh run_train_loc.sh
       cd ..
       ```
    - **Retrain Repair Model**
       ```bash
       cd VQM
       sh run_pretrain.sh
       sh run_train.sh
       sh run_test.sh
       cd ..
       ```
- **VulRepair**

    - **Inference**
        ```bash
        cd baselines/VulRepair/saved_models/checkpoint-best-loss
        sh download_models.sh
        cd ../..
        sh run_test.sh
        cd ../..
        ```

    - **Retrain**
       ```bash
       cd baselines/VulRepair
       sh run_pretrain.sh
       sh run_train.sh
       sh run_test.sh
       cd ../..
       ```
 - **TFix**

    - **Inference**
        ```bash
        cd baselines/TFix/saved_models/checkpoint-best-loss
        sh download_models.sh
        cd ../..
        sh run_test.sh
        cd ../..
        ```

    - **Retrain**
       ```bash
       cd baselines/TFix
       sh run_pretrain.sh
       sh run_train.sh
       sh run_test.sh
       cd ../..
       ```
  - **GraphCodeBERT**

    - **Inference**
        ```bash
        cd baselines/GraphCodeBERT/saved_models/checkpoint-best-loss
        sh download_models.sh
        cd ../..
        sh run_test.sh
        cd ../..
        ```

    - **Retrain**
       ```bash
       cd baselines/GraphCodeBERT
       sh run_pretrain.sh
       sh run_train.sh
       sh run_test.sh
       cd ../..
       ```
  - **CodeBERT**

    - **Inference**
        ```bash
        cd baselines/CodeBERT/saved_models/checkpoint-best-loss
        sh download_models.sh
        cd ../..
        sh run_test.sh
        cd ../..
        ```

    - **Retrain**
       ```bash
       cd baselines/CodeBERT
       sh run_pretrain.sh
       sh run_train.sh
       sh run_test.sh
       cd ../..
       ```
 - **VRepair**

    - **Inference**
        ```bash
        cd baselines/VRepair/saved_models/checkpoint-best-loss
        sh download_models.sh
        cd ../..
        sh run_test.sh
        cd ../..
        ```

    - **Retrain**
       ```bash
       cd baselines/VRepair
       sh run_pretrain.sh
       sh run_train.sh
       sh run_test.sh
       cd ../..
       ```
  - **SequenceR**

    - **Inference**
        ```bash
        cd baselines/SequenceR/saved_models/checkpoint-best-loss
        sh download_models.sh
        cd ../..
        sh run_test.sh
        cd ../..
        ```

    - **Retrain**
       ```bash
       cd baselines/SequenceR
       sh run_pretrain.sh
       sh run_train.sh
       sh run_test.sh
       cd ../..
       ```
#### Reproduce Section 4 - RQ2
- **Only denoising**

    - **Inference**
        ```bash
        cd AutoVF/AutoVF-main/ablation_model/only_denoising
        sh run_pretrain_loc.sh
        sh run_train_loc.sh
        cd ..
        ```

    - **Retrain**
       ```bash
      cd AutoVF/AutoVF-main/ablation_model/only_denoising
      sh run_pretrain.sh
      sh run_train.sh
      sh run_test.sh
      cd ..
       ```
- **Only mixup**

    - **Inference**
        ```bash
        cd AutoVF/AutoVF-main/ablation_model/only_mixup
        sh run_pretrain_loc.sh
        sh run_train_loc.sh
        cd ..
        ```

    - **Retrain**
       ```bash
      cd AutoVF/AutoVF-main/ablation_model/only_mixup
      sh run_pretrain.sh
      sh run_train.sh
      sh run_test.sh
      cd ..
       ```
#### Reproduce Section 4 - RQ3
- **co-teaching**

    - **Inference**
        ```bash
        cd AutoVF/AutoVF-main/ablation_model/noise_label/co-teaching
        sh run_pretrain_loc.sh
        sh run_train_loc.sh
        cd ..
        ```

    - **Retrain**
       ```bash
      cd AutoVF/AutoVF-main/ablation_model/noise_label/co-teaching
      sh run_pretrain.sh
      sh run_train.sh
      sh run_test.sh
      cd ..
       ```
- **co-teaching+**

    - **Inference**
        ```bash
        cd AutoVF/AutoVF-main/ablation_model/noise_label/co-teaching+
        sh run_pretrain_loc.sh
        sh run_train_loc.sh
        cd ..
        ```

    - **Retrain**
       ```bash
      cd AutoVF/AutoVF-main/ablation_model/noise_label/co-teaching+
      sh run_pretrain.sh
      sh run_train.sh
      sh run_test.sh
      cd ..
       ```
- **jocor**

    - **Inference**
        ```bash
        cd AutoVF/AutoVF-main/ablation_model/noise_label/jocor
        sh run_pretrain_loc.sh
        sh run_train_loc.sh
        cd ..
        ```

    - **Retrain**
       ```bash
      cd AutoVF/AutoVF-main/ablation_model/noise_label/jocor
      sh run_pretrain.sh
      sh run_train.sh
      sh run_test.sh
      cd ..
       ```
#### Reproduce Section 4 - RQ4
Repeat experiments with different interpolation parameters

 - **Inference**
    ```bash
    cd AutoVF/AutoVF-main/Parameter/Interpolation Parameter
    sh run_pretrain_loc.sh
    sh run_train_loc.sh
    cd ..
    ```

- **Retrain**
    ```bash
    cd AutoVF/AutoVF-main/Parameter/Interpolation Parameter
    sh run_pretrain.sh
    sh run_train.sh
    sh run_test.sh
    cd ..
    ```
#### Reproduce Section 4 - RQ5
Repeat experiments with different compensation parameters

- **Inference**
    ```bash
    cd AutoVF/AutoVF-main/Parameter/Compensation Parameter
    sh run_pretrain_loc.sh
    sh run_train_loc.sh
    cd ..
    ```

- **Retrain**
    ```bash
    cd AutoVF/AutoVF-main/Parameter/Compensation Parameter
    sh run_pretrain.sh
    sh run_train.sh
    sh run_test.sh
    cd ..
    ```
