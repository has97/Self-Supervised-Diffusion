# Self-Supervised-Diffusion
## Steps to generate the images
First change the line 29,36,37 imagenet_aug.py to train path of the required dataset train path. Then run the following command
```shell
python test.py
```
Then the images will be generated in folder named diffimage which is outside the current folder
This repo is build on solo repositry: https://github.com/vturrisi/solo-learn
Make the changes in the yaml file corresponding to model in scripts/pretrain/imagenet100 folder.
To run the model use the following command.
```shell
python3 main_linear.py --config-path "scripts/linear/imagenet-100/" --config-name "byol.yaml"
```
where the yaml file is to be replaced by the corresponding model file.
