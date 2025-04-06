# VIPeR

### [Paper](https://ieeexplore.ieee.org/document/10873856) | [arXiv](https://arxiv.org/abs/2407.21416) | [Website](https://x-mak.github.io/VIPeR/)

> SLC<sup>2</sup>-SLAM: Semantic-guided Loop Closure using Shared Latent Code for NeRF SLAM <br />
> [Yuhang Ming](https://yuhangming.github.io/), [Minyang Xu](), [Xingrui Yang](), [Weicai Ye](), [Weihan Wang](), [Yong Peng](), [Weichen Dai](), [Wanzeng Kong](https://faculty.hdu.edu.cn/zzb/kwz/main.htm)<br />
> RA-L 2025

This repo contains the source code for paper:

"[VIPeR: Visual Incremental Place Recognition with Adaptive Mining and Continual Learning.](https://arxiv.org/pdf/2407.21416)"IEEE Robotics and Automation Letters(RAL)

## Usage

**Installation**

1. Clone this repo:

```
git clone https://github.com/X-mak/VIPeR_code.git
cd VIPeR_code
```

2. Install dependencies

```
conda env create --file environment.yaml
conda activate viper
```

**Data**

We used the following subsets of datasets in our expriments:

 - [TartanAir](https://theairlab.org/tartanair-dataset/), download with [tartanair_tools](https://github.com/castacks/tartanair_tools)
   - Train/Test: `abandonedfactory_night`, `carwelding`, `neighborhood`, `office2`, `westerndesert`;
 - [RobotCar](https://robotcar-dataset.robots.ox.ac.uk/), download with [RobotCarDataset-Scraper](https://github.com/mttgdd/RobotCarDataset-Scraper)
   - Train: `2014-11-28-12-07-13`, `2014-12-10-18-10-50`, `2014-12-16-09-14-09`; 
   - Test: `2014-06-24-14-47-45`, `2014-12-05-15-42-07`, `2014-12-16-18-44-24`;
 - [Nordland](https://webdiis.unizar.es/~jmfacil/pr-nordland/), download with [gdown](https://github.com/wkentaro/gdown) from [Google Drive](https://drive.google.com/drive/folders/1SmrDOeUgBnJbpW187VFWxGjS7XdbZK5t)
   - Train/Test: All four seasons with recommended splits.

The datasets are aranged as follows:

```
$DATASET_ROOT/
├── tartanair/
│   ├── abandonedfactory_night/
|   |   ├── Easy/
|   |   |   └── ...
│   │   └── Hard/
│   │       └── ...
│   └── ...
├── robotcar/
│   ├── train/
│   │   ├── 2014-11-28-12-07-13/
│   │   └── ...
│   └── test/
│       ├── 2014-06-24-14-47-45/
│       └── ...
└── nordland/
    ├── train/
    │   ├── fall_images_train/
    │   └── ...
    └── test/
        ├── fall_images_test/
        └── ...
```

**Command**

The following command trains the model with the specified method on TartanAir with default configuration and evaluate the performance:

```sh
$ python main.py --method <finetune/airloop/viper> 
```

Extra options*:

* `--out-dir <DIR>`: output directory for model checkpoints and importance weights.
* `--memory-path <DIR>`: output directory for memory file.
* `--skip-eval`: perform training only.
* `--skip-train`: perform evaluation only.
* `--cross`: launch a multi-dataset environment.
* `--dataset <tartanair/robotcar/nordland>`: dataset to use. If the "cross" option is added, multiple datasets can be specified in the parameters, and the model will be trained in the order of the list. For example: `--dataset tartanair,robotcar,nordland`.
* `--log-dir <DIR>`: Tensorboard `logdir`.
* `--backbone <vgg/dinov2>`: the backbone used by the model.
* `--aggregator <gem/vlad>`: the aggregator used by the model.

See main_single.py for more settings.

**Model**

We also provide pre-trained models [here](https://drive.google.com/drive/folders/1HjR3O8qwlRWcdqMQbP3AleIkFnV5H7G2?usp=drive_link), which include three datasets (Nordland, RobotCar, TartanAir) and models trained in various environments for each dataset. The backbone of these models is VGG-19, and the aggregator is NetVLAD.

## Citation

If you find our work helpful, please consider citing:

```
@ARTICLE{10873856,
  author={Ming, Yuhang and Xu, Minyang and Yang, Xingrui and Ye, Weicai and Wang, Weihan and Peng, Yong and Dai, Weichen and Kong, Wanzeng},
  journal={IEEE Robotics and Automation Letters}, 
  title={VIPeR: Visual Incremental Place Recognition with Adaptive Mining and Continual Learning}, 
  year={2025},
  volume={10},
  number={3},
  pages={3038-3045},
  doi={10.1109/LRA.2025.3539093}}
```

