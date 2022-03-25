# LE-GAN: Unsupervised Low-light Image Enhancement Network using Attention Module and Identity Invariant Loss

​	This is Paired Normal/Low-light Images (PNLI) dataset and Pytorch implementation of LE-GAN: Unsupervised Low-light Image Enhancement Network using Attention Module and Identity Invariant Loss in KBS(Knowledge-Based Systems), 2022, by Ying Fu, Yang Hong, Linwei Chen, Shaodi You.

------

[**Paper**](https://www.sciencedirect.com/science/article/abs/pii/S0950705121011151)

------



## Representitive Results

![](https://cdn.jsdelivr.net/gh/MUYIio/CDN@2.3/Images/Paper/3.png)



## Overal Architecture

![](https://cdn.jsdelivr.net/gh/MUYIio/CDN@2.3/Images/Paper/4.png)

## Environment Preparing

The code is tested on Python 3.7, PyTorch 1.7.0, TorchVision 0.8.1, but lower versions are also likely to work.



## Paired Normal/low-light Images (PNLI) Dataset

**Images for PNLI**

1. **[Normal-light data](https://pan.baidu.com/s/1Kvis8n-EeTnfB1QtNbPuRA)** (Ground-truth). (Extraction Code: 2022)

2. **[Low-light data](https://pan.baidu.com/s/1nKFmiYAcbZVqegqnRP0ZPQ)**. (Extraction Code: 2022)

**Tips:**

1. We provide all files in **[Baidu Drive]**, and the extraction code of all files is “2022”.

2. Note that each low-light image correspond to one normal-light image as Ground Truth.

3. We name all images with a purely numeric number, paired long/short-exposure images file names of the same format are corresponding(low-light image file name = normal light image file name+1). For example, for “1.JPG”, the file name of the corresponding low-light image is “2. JPG”.



## Citation

If you use our dataset or code for research, please ensure that you cite our paper:

Ying Fu, Yang Hong, Linwei Chen, and Shaodi You, "LE-GAN: Unsupervised low-light image enhancement network using attention module and identity invariant loss", in Knowledge-Based Systems, 2022, 240: 108010.

```
@article{fu2022gan,
  title={LE-GAN: Unsupervised low-light image enhancement network using attention module and identity invariant loss},
  author={Fu, Ying and Hong, Yang and Chen, Linwei and You, Shaodi},
  journal={Knowledge-Based Systems},
  volume={240},
  pages={108010},
  year={2022},
  publisher={Elsevier}
}

```

​          

## Questions

If you have any additional questions, please email to hongyang@bit.edu.cn
