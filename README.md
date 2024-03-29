
# ENHANCING THE DOMAIN ROBUSTNESS OF SELF-SUPERVISED PRE-TRAINING WITH SYNTHETIC IMAGES

[![Conference](https://img.shields.io/badge/ICASSP-2024-4b44ce)](https://2024.ieeeicassp.org/)

<a href="https://www.python.org"><img alt="Python" src="https://img.shields.io/badge/-Python_3.7-blue?logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch_1.10-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning_2.1.2-792ee5?logo=pytorchlightning&logoColor=white"></a>

## Steps to generate the images
First change the line 29,36,37 imagenet_aug.py to train path of the required dataset train path. Then run the following command
```shell
python test.py
```
Then the images will be generated in folder named diffimage which is outside the current folder
This repo is build on solo repositry: https://github.com/vturrisi/solo-learn
## Steps to setup the environment
- Go to the repositry and clone https://github.com/vturrisi/solo-learn and then enter the solo-learn directory.
- Then create a new environment using the following command 
```shell
conda create -n ssl-diff python=3.7
conda activate ssl-diff
```
- Then install all the required libraries using the pip command in the repositry https://github.com/vturrisi/solo-learn.
- Then enter to this repo.
## Changing the YAML file for the required datasets
- Go to the scripts/pretrain/imagenet-100/model_name.yaml and change the train_path , val_path [line](https://github.com/has97/Self-Supervised-Diffusion/blob/04e2315dad01366b5100d2bc4309501968bdc1e6/scripts/pretrain/imagenet-100/byol.yaml#L27) to the required directory of training images and validation images.
- Then run the following command for training and linear evaluation.

To train the model use the following command.
```shell
python3 main_pretrain.py --config-path "scripts/pretrain/imagenet-100/" --config-name "byol.yaml"
```
where the yaml file is to be replaced by the corresponding model file. 

To evaluate the model use the following command.
```shell
python3 main_linear.py --config-path "scripts/linear/imagenet-100/" --config-name "byol.yaml"
```
