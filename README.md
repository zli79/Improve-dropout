# Improve-dropout
This repository contains instroduction, instruction, models, prototxt and logs file along experiments for Improved-dropout introduced in the paper ["Improved Dropout for Shallow and Deep Learning"](http://papers.nips.cc/paper/6561-improved-dropout-for-shallow-and-deep-learning.pdf) by Zhe Li, Boqing Gong, Tianbao Yang

### Citing Improved-Dropout
If you are using improved dropout, please cite as following:

        @inproceedings{li2016improved,
          title={Improved dropout for shallow and deep learning},
          author={Li, Zhe and Gong, Boqing and Yang, Tianbao},
          booktitle={Advances in Neural Information Processing Systems},
          pages={2523--2531},
          year={2016}
        }
# Contents:
1. [Introduction](#Introduction)
2. [Explanation Source Code](#Explanation-Source-Code)
3. [Experiments on CIFAR100](#Experiments-on-CIFAR100)
4. [Summary](#Summary)

## Introduction
We propose to use multinomial sampling for dropout, i.e., sampling features or neurons according to a multinomial distribution with different probabilities for different features/neurons. To exhibit the optimal dropout probabilities, we analyze the shallow learning with multinomial dropout and establish the risk bound for stochastic optimization. By minimizing a sampling dependent factor in the risk bound, we obtain a distribution-dependent dropout with sampling probabilities dependent on the second order statistics of the data distribution. To tackle the issue of evolving distribution of neurons in deep learning, we propose an effecient adaptive dropout (named evolutional dropout) that computes the sampling probabilities on-the-fly from a mini-batch of examples. 

## Explanation Source Code
We conducted experiments based on [CudaConvNet](https://github.com/akrizhevsky/cuda-convnet2), in which we implemented the functionality of improve dropout (sampling from multinomial distribution). 
