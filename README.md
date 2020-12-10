## Impersonator++
**Liquid Warping GAN with Attention: A Unified Framework for Human Image Synthesis**, including 
human motion imitation, appearance transfer, and novel view synthesis. Currently the paper is under review of 
IEEE TPAMI. It is an extension of our previous ICCV project [impersonator](https://github.com/svip-lab/impersonator), 
and it has a more powerful ability in generalization and produces higher-resolution results  (512 x 512, 1024 x 1024) than the previous ICCV version.

|  🧾 Colab Notebook  |   📑 Paper    | 📱 Website | 📂 Dataset | 💡 Bilibili | ✒ Forum |
  |------------|-------------|-----------|-----------|-----------|-----------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bwUnj-9NnJA2EMr7eWO4I45UuBtKudg_?usp=sharing) | [paper](https://arxiv.org/pdf/2011.09055.pdf) | [website](https://www.impersonator.org/work/impersonator-plus-plus.html) | [Dataset](https://svip-lab.github.io/dataset/iPER_dataset.html) | [bilibili](https://space.bilibili.com/1018066133) | [Forum](https://discuss.impersonator.org/)|


![](https://www.impersonator.org/images/motion_results.png)


## Update News
- [x] 12/06/2020, iPERCore-0.1, all the base codes. The motion imitation scripts.
- [x] 12/10/2020, iPERCore-0.1.1, supports Windows.

See the details of developing [logs](./docs/dev_logs.md).


## Installation
See more details, including system dependencies, python requirements and setups in [install.md](./docs/install.md).
Please follows the instructions in [install.md](./docs/install.md) to install this firstly.

## Run demos

### 1. Run on Google Colab 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bwUnj-9NnJA2EMr7eWO4I45UuBtKudg_?usp=sharing)


### 2. Run with Console (scripts) mode
See [scripts_runner](./docs/scripts_runner.md) for more details.

### 3. Run with GUI mode
Coming soon!

## Citation
```
@misc{liu2020liquid,
      title={Liquid Warping GAN with Attention: A Unified Framework for Human Image Synthesis}, 
      author={Wen Liu and Zhixin Piao, Zhi Tu, Wenhan Luo, Lin Ma and Shenghua Gao},
      year={2020},
      eprint={2011.09055},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@InProceedings{lwb2019,
    title={Liquid Warping GAN: A Unified Framework for Human Motion Imitation, Appearance Transfer and Novel View Synthesis},
    author={Wen Liu and Zhixin Piao, Min Jie, Wenhan Luo, Lin Ma and Shenghua Gao},
    booktitle={The IEEE International Conference on Computer Vision (ICCV)},
    year={2019}
}
```



