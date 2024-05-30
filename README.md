# Rethinking Class Incremental Learning from a Dynamic Imbalanced Learning Perspective

---

## Introduction

This code is based on [PyCIL](https://github.com/G-U-N/PyCIL) with  modifications to details such as log output. All the original logs of the experiments mentioned in the paper are located in the `logs/`. The code for our proposed UPCL is located in `models/upcl.py`.

If you use our method or code in your research, please consider citing the paper as follows:

```
@article{wang2024rethinking,
  title={Rethinking Class-Incremental Learning from a Dynamic Imbalanced Learning Perspective},
  author={Wang, Leyuan and Xiang, Liuyu and Wang, Yunlong and Wu, Huijia and He, Zhaofeng},
  journal={arXiv preprint arXiv:2405.15157},
  year={2024}
}
```

### Dependencies

1. [torch 1.81](https://github.com/pytorch/pytorch)
2. [torchvision 0.6.0](https://github.com/pytorch/vision)
3. [tqdm](https://github.com/tqdm/tqdm)
4. [numpy](https://github.com/numpy/numpy)
5. [scipy](https://github.com/scipy/scipy)

### Run experiment

1. Edit the `[MODEL NAME].json` file for global settings.
2. Edit the hyper-parameters in the corresponding `[MODEL NAME].py` file (e.g., `models/icarl.py`).
3. Run: `python main.py --config=./exps/[MODEL NAME].json`
4. `hyper-parameters`

When using PyCIL, you can edit the global parameters and algorithm-specific hyper-parameter in the corresponding json file.

These parameters include:

- **memory-size**: The total exemplar number in the incremental learning process. Assuming there are $K$ classes at the current stage, the model will preserve $\left[\frac{memory\_size}{K}\right]$ exemplar per class.
- **init-cls**: The number of classes in the first incremental stage. Since there are different settings in CIL with a different number of classes in the first stage, our framework enables different choices to define the initial stage.
- **increment**: The number of classes in each incremental stage $i$, $i$ > 1. By default, the number of classes per incremental stage is equivalent per stage.
- **convnet-type**: The backbone network for the incremental model.
- **seed**: The random seed adopted for shuffling the class order. According to the benchmark setting, it is set to 1993 by default.


### Datasets

We have implemented the pre-processing of `CIFAR100`, `imagenet100,` `Tinyimagenet,` and `imagenet1000`. When training on `CIFAR100`, this framework will automatically download it.  When training on `imagenet100/1000` and `Tinyimagenet`, you should specify the folder of your dataset in `utils/data.py`.

```python
    def download_data(self):
        assert 0,"You should specify the folder of your dataset"
        train_dir = '[DATA-PATH]/train/'
        test_dir = '[DATA-PATH]/val/'
```
