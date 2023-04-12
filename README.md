# Impersonator++

### Update News
- [x] 15/04/2021, iPERCore-0.2.0, including
  
   * Add [Training](./docs/train_details.md)
   
   * Add [Novel View Synthesis](https://github.com/iPERDance/iPERCore/blob/main/docs/scripts_runner.md#run-novel-view-synthesis)
   
   * Add [Motion Imitation with bullet-time effects](https://github.com/iPERDance/iPERCore/blob/main/docs/scripts_runner.md#run-motion-imitation-with-bullet-time-effect)
   
   * Add [Motion Imitation with multi-view outputs](https://github.com/iPERDance/iPERCore/blob/main/docs/scripts_runner.md#run-motion-imitation-with-multi-view-outputs)
   
   * Add [Appearance Transfer](https://github.com/iPERDance/iPERCore/blob/main/docs/scripts_runner.md#run-human-appearance-transfer)
   
   * Add [A Unified synthesizer: Motion Imitation + Appearance Transfer + Novel View Synthesis](https://github.com/iPERDance/iPERCore/blob/main/docs/scripts_runner.md#human-appearance-transfer-with-motion-imitation-and-novel-view-synthesis)
   
   * Update torch 1.8+ with RTX30+ GPUs.
  
[comment]: <> (- [x] 12/20/2020, A precompiled version on Windows has been released! [[Usage]]&#40;https://github.com/iPERDance/iPERCore/wiki/How-to-use-the-released-version-on-windows%3F&#41;)
- [x] 12/10/2020, iPERCore-0.1.1, supports Windows.
- [x] 12/06/2020, iPERCore-0.1, all the base codes. The motion imitation scripts.


See the details of developing [logs](./docs/dev_logs.md).

**Liquid Warping GAN with Attention: A Unified Framework for Human Image Synthesis**, including 
human motion imitation, appearance transfer, and novel view synthesis. Currently the paper is under review of 
IEEE TPAMI. It is an extension of our previous ICCV project [impersonator](https://github.com/svip-lab/impersonator), 
and it has a more powerful ability in generalization and produces higher-resolution results  (512 x 512, 1024 x 1024) than the previous ICCV version.

<!-- |  🧾 Colab Notebook  | Released (Windows)  |   📑 Paper    | 📱 Website | 📂 Dataset | 💡 Bilibili | ✒ Forum |
 :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bwUnj-9NnJA2EMr7eWO4I45UuBtKudg_?usp=sharing) | [[Usage]](https://github.com/iPERDance/iPERCore/wiki/How-to-use-the-released-version-on-windows%3F) | [paper](https://arxiv.org/pdf/2011.09055.pdf) | [website](https://www.impersonator.org/work/impersonator-plus-plus.html) | [Dataset](https://svip-lab.github.io/dataset/iPER_dataset.html) | [bilibili](https://space.bilibili.com/1018066133) | [Forum](https://iperdance.github.io/work/impersonator-plus-plus.html)| -->

|  🧾 Colab Notebook |   📑 Paper    | 📱 Website | 📂 Dataset | 💡 Bilibili |
 :-: | :-: | :-: | :-: | :-: |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bwUnj-9NnJA2EMr7eWO4I45UuBtKudg_?usp=sharing) | [paper](https://arxiv.org/pdf/2011.09055.pdf) | [website](https://iperdance.github.io/work/impersonator-plus-plus.html) | [Dataset](https://svip-lab.github.io/dataset/iPER_dataset.html) | [bilibili](https://space.bilibili.com/1018066133) |


![](https://iperdance.github.io/images/motion_results.png)


## Installation
See more details, including system dependencies, python requirements and setups in [install.md](./docs/install.md).
Please follows the instructions in [install.md](./docs/install.md) to install this firstly.

**Notice that `imags_size=512` need at least 9.8GB GPU memory.** if you are using a middle-level GPU(e.g. RTX 2060), you should change the `image_size` to 384 or 256. The following table can be used as a reference:

| image_size | preprocess | personalize | run_imitator | recommended gpu                    |
| ---------- | ---------- | ----------- | ------------ | ---------------------------------- |
| 256x256    | 3.1 GB     | 4.3 GB      | 1.1 GB       | RTX 2060 / RTX 2070                |
| 384x384    | 3.1 GB     | 7.9 GB      | 1.5 GB       | GTX 1080Ti / RTX 2080Ti / Titan Xp |
| 512x512    | 3.1 GB     | 9.8 GB      | 2 GB         | GTX 1080Ti / RTX 2080Ti / Titan Xp |
| 1024x1024  | 3.1 GB     | 20 GB       | -            | RTX Titan / P40 / V100 32G         |


## Run demos

### 1. Run on Google Colab 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bwUnj-9NnJA2EMr7eWO4I45UuBtKudg_?usp=sharing)


### 2. Run with Console (scripts) mode
See [scripts_runner](./docs/scripts_runner.md) for more details.

## Citation
```
@article{liu2021liquid,
  title={Liquid warping GAN with attention: A unified framework for human image synthesis},
  author={Liu, Wen and Piao, Zhixin and Tu, Zhi and Luo, Wenhan and Ma, Lin and Gao, Shenghua},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021},
  publisher={IEEE}
}

@InProceedings{lwb2019,
    title={Liquid Warping GAN: A Unified Framework for Human Motion Imitation, Appearance Transfer and Novel View Synthesis},
    author={Wen Liu and Zhixin Piao, Min Jie, Wenhan Luo, Lin Ma and Shenghua Gao},
    booktitle={The IEEE International Conference on Computer Vision (ICCV)},
    year={2019}
}
```



