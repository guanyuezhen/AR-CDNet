# <p align=center>`Towards Accurate and Reliable Change Detection of Remote Sensing Images via Knowledge Review and Online Uncertainty Estimation (Under Review)`</p>

This repository contains simple python implementation of our paper [AR-CDNet](https://arxiv.org/abs/2305.19513).

### 1. Overview

<p align="center">
    <img width=500 src="assest/AR-CDNet.jpg"/> <br />
</p>

A framework of the proposed AR-CDNet. Initially, the bi-temporal images pass through a shared feature extractor to obtain bi-temporal features, and then multi-level temporal difference features are obtained through the TDE. The OUE branch estimates pixel-wise uncertainty supervised by the diversity between predicted change maps and corresponding ground truth in the training process. KRMs fully explore the multi-level temporal difference knowledge. Finally, the multi-level temporal difference features and uncertainty-aware features obtained from the OUE branch are aggregated to generate the final change maps. <br>

### 2. Usage
+ Prepare the data:
    - Download datasets [LEVIR](https://justchenhao.github.io/LEVIR/), and [BCDD](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html).
    - Crop LEVIR and BCDD datasets into 512x512 patches. The pre-processed LEVIR and BCDD datasets can be obtained from [BCDD_512x512](https://drive.google.com/file/d/1VrdQ-rxoGVM_8ecA-ObO0u-O8rSTpSHA/view?usp=sharing), [BCDD_512x512](https://drive.google.com/file/d/1VrdQ-rxoGVM_8ecA-ObO0u-O8rSTpSHA/view?usp=sharing).
    - Generate list file as `ls -R ./label/* > test.txt`
    - Prepare datasets into the following structure and set their path in `train.py` and `test.py`
    ```
    ├─Train
        ├─A        ...jpg/png
        ├─B        ...jpg/png
        ├─label    ...jpg/png
        └─list     ...txt
    ├─Val
        ├─A
        ├─B
        ├─label
        └─list
    ├─Test
        ├─A
        ├─B
        ├─label
        └─list
    ```

+ Prerequisites for Python:
    - Creating a virtual environment in the terminal: `conda create -n AR-CDNet python=3.8`
    - Installing necessary packages: `pip install -r requirements.txt `

+ Train/Test
    - `sh train.sh`
    - `sh test.sh`

### 3. Citation

Please cite our paper if you find the work useful:

    @article{Li_2023_MSL-MKC,
            title={Towards Accurate and Reliable Change Detection of Remote Sensing Images via Knowledge Review and Online Uncertainty Estimation},
            author={Li, Zhenglai and Tang, Chang and Li, Xianju and Xie, Weiying and Sun, Kun and Zhu, Xinzhong},
            journal={arXiv preprint arXiv:2305.19513},
            year={2023}
        }
