# Unsupervised Contrastive Learning Based Single Domain Generalization Method for Intelligent Bearing Fault Diagnosis

Qiang Wu, Yue Ma, Zhixi Feng, Shuyuan Yang and Hao Hu, "[Unsupervised Contrastive Learning Based Single Domain Generalization Method for Intelligent Bearing Fault Diagnosis](https://ieeexplore.ieee.org/document/10785570)", IEEE Sensors Journal.

## Abstract

<p align="center">
<img src="https://github.com/WuQiangXDU/" width="600" class="center">
</p>

In the field of fault diagnosis, an increasing number of domain generalization methods are being employed to address domain shift issues. The vast majority of these methods focus on learning domain-invariant features from multiple source domains, with very few considering the more realistic scenario of a single source domain. Furthermore, there is a lack of work that achieves single-domain generalization through unsupervised means. Therefore, in this paper we introduce a data augmentation method for frequency-domain signals called Multi-Amplitude Random Spectrum (MARS), which randomly adjusts the amplitude of each point in the spectrum to generate multiple pseudo-target domain samples from a single source domain sample. Then we combine MARS with unsupervised contrastive learning to bring the pseudo target domain samples closer to the source domain samples in the feature space, which enables generalization to unknown target domains since the pseudo target domain samples contain potentially true target domain samples as much as possible. Unsupervised single-domain generalization intelligent fault diagnosis can thus be achieved. Extensive experiments on three datasets demonstrate effectiveness of the proposed method.


## Platform

- NVIDIA 4090 24GB GPU, PyTorch

## Usage

1. Prepare Data. You can obtain the datasets from [Baidu Drive](https://pan.baidu.com/s/1fiXX5shuTl6C34agHo0w5A?pwd=a3q2), then place the downloaded data in the folder ````./Datasets````.

2. Creating the environment with conda.

````
conda create -n uclsdg python=3.8
conda activate uclsdg
pip install -r requirements.txt
conda install pytorch==1.12.0 cudatoolkit=11.6
````

3. Train and evaluate model:
````
python train_and_tset.py
````

4. The results of the experiment are stored in ````results````.