# Recursive Re-parameterization for Better Training

Anonymous code submission for our paper "Recursive Re-parameterization for Better Training".

This code is implemented with [Pytorch](https://github.com/pytorch/pytorch). We thank every reviewer for their hard work.

*Note: due to the submission policies, this code will not be modified after the submission deadline.*


[**Abstract**](#abstract) | [**Requirements**](#requirements) | [**Data Preparation**](#data-preparation) | [**Training**](#training) | [**Pre-trained Weights**](#pre-trained-weights)

<p>
<img src="https://img.shields.io/badge/Python-%3E%3D3.7-blue">
<img src="https://img.shields.io/badge/PyTorch-1.9-informational">
</p>

---


<br>



## Abstract

Network re-parameterization (Rep) merges convolutional branches into a mathematically equivalent single layer before model deployment, to reduce the memory footprint and latency at run time. We argue that an overlooked benefit of Rep is its effects on model accuracy enhancement. Through extensive experiments, we find it is surprisingly capable of improving a model's accuracy, if used correctly. Based on the findings, we propose a simple yet effective approach named Recursive-Rep to obtain considerable accuracy gains without increasing the FLOPs or number of parameters. Specifically, in the training stage, the network goes through cycles of re-parameterization, expansion, inversion, and backpropagation operations. Such a method is generalized and naturally compatible with all Rep structures. We explore possible explanations for the effectiveness of our work, with corresponding theoretical proof. Our approach improves the performance of baselines by 6.38% on the CIFAR-100 dataset, and by 2.80% on ImageNet, achieving state-of-the-art performance in the Rep field.


## Requirements

To run our code, `python>=3.7` and `pytorch>=1.9` are required. Other versions of `PyTorch` may also work well, but there are potential API differences that can cause warnings to be generated.

Other required packages are listed in the `requirements.txt` file. You can simply install these dependencies by:

```bash
pip install -r requirements.txt
```

Then, set the `$PYTHONPATH` environment variable before running this code:

```bash
export PYTHONPATH=$PYTHONPATH:/Path/to/This/Code
```

(Optional) Set the visible GPU devices:

```bash
export CUDA_VISIBLE_DEVICES=0
```


## Data Preparation

The dataloaders of our code read the dataset files from `$CODE_PATH/data/$DATASET_NAME` by default, and we use lowercase filenames and remove the hyphens. For example, files for CIFAR-10 should be placed (or auto downloaded) under `$CODE_PATH/data/cifar10` directory.

Our method can load pre-trained weights for faster training. To achieve this, place the weights file in the `$CODE_PATH/weights` folder, and modify the configuration files correspondingly.


## Training

Our code reads configuration files from the command line, and can be overriddden by manually adding or modifying. We list several examples for running our code as follows:

#### Baselines

**Normal training** 

```bash
python tools/train/CODE.py --cfg configs/CONFIG_FILE.yaml
```


**Knowledge Distillation (KD) training** 

```bash
python tools/kd/CODE.py --cfg configs/CONFIG_FILE.yaml
```


Examples:

```bash
python tools/train/train_repvgg_cifar.py --cfg configs/rep/rep_c100_normal.yaml
```

```bash
python tools/kd/kd_dbb_cifar.py --cfg configs/dbb/dbb_c10_normal.yaml
```

#### Our Approach

```bash
python tools/cycle/CODE.py --cfg configs/CONFIG_FILE.yaml
```


Examples:

```bash
python tools/cycle/cycle_repvgg_cifar.py --cfg configs/rep/rep_c100_cycle.yaml
```

```bash
python tools/cycle/cycle_dbb_cifar.py --cfg configs/ac/acb_c10_cycle.yaml
```

```bash
python tools/cycle/cycle_dbb_img.py --cfg configs/dbb_IMG_cycle.yaml
```

Other cases can be run by modifying the configuration files in the `configs` folder.


## Pre-trained Weights

The released checkpoint files of our method are as follows:


|CIFAR-10|CIFAR-100|ImageNet|
|---|---|---|
|[RepVGG-A1](http://cdn.thrase.cn/cvpr23/c10_repvgg_A1.pyth)|[RepVGG-A1](http://cdn.thrase.cn/cvpr23/c100_repvgg_A1.pyth)|[MobileNet](http://cdn.thrase.cn/cvpr23/mb.pth.tar)|
|[RepVGG-B1](http://cdn.thrase.cn/cvpr23/c10_repvgg_B1.pyth)|[RepVGG-B1](http://cdn.thrase.cn/cvpr23/c100_repvgg_B1.pyth)|[ResNet18](http://cdn.thrase.cn/cvpr23/r18.pth.tar)|
|[RepVGG-B3](http://cdn.thrase.cn/cvpr23/c10_repvgg_B3.pyth)|[RepVGG-B3](http://cdn.thrase.cn/cvpr23/c100_repvgg_B3.pyth)|[ResNet34](http://cdn.thrase.cn/cvpr23/r34.pth.tar)|
|[ResNet18-DBB](http://cdn.thrase.cn/cvpr23/c10_dbb_r18.pyth)|[ResNet18-DBB](http://cdn.thrase.cn/cvpr23/c100_dbb_r18.pyth)|[ResNet50](http://cdn.thrase.cn/cvpr23/r50.pth.tar)|
|[ResNet18-ACNet](http://cdn.thrase.cn/cvpr23/c10_ac_r18.pyth)|[ResNet18-ACNet](http://cdn.thrase.cn/cvpr23/c100_ac_r18.pyth)|[RepVGG-B3](http://cdn.thrase.cn/cvpr23/repvgg_b3.pth.tar)|


The checkpoint file contains:

```
CheckpointFile (dict)
├─── epoch (int)
└─── model_state (dict)
```