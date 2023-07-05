
<p align="center">
  <img width="100%" src="https://github.com/bxiang233/PanopticSegForLargeScalePointCloud/blob/main/Docs/instance%20segmentation%20visualization%20new.png" />
</p>

# Set up environment

The framework used in this code is torchpoints-3d, so generally the installation instructions for torchpoints-3d can follow the official ones: 

https://torch-points3d.readthedocs.io/en/latest/

https://github.com/torch-points3d/torch-points3d

Here are two detailed examples for installation worked on our local computers for your reference:

### Example 1 of installation

Specs local computer: Ubuntu 22.04, 64-bit, CUDA version 11.7 -> but CUDA is backwards compatible, so here we used CUDA 11.1 for all libraries installed.

Commands in terminal using miniconda:
```bash
conda create -n treeins_env_local python=3.8

conda activate treeins_env_local

conda install pytorch=1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch -c nvidia

pip install numpy==1.19.5

conda install openblas-devel -c anaconda

export CUDA_HOME=/usr/local/cuda-11

pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" –install-option="--blas=openblas"

#CHECK IF TORCH AND MINKOWSKI ENGINE WORK AS EXPECTED:
(treeins_env_local) : python
Python 3.8.13 (default, #DATE#, #TIME#)
[GCC 7.5.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.cuda.is_available()
True
>>> import MinkowskiEngine
>>> exit()
#CHECK FINISHED

pip install torch-scatter==2.0.8 -f https://data.pyg.org/whl/torch-1.9.0+cu111.html

pip install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.9.0+cu111.html

pip install torch-geometric==1.7.2

#We got the file requirements.txt from here: https://github.com/nicolas-chaulet/torch-points3d/blob/master/requirements.txt but deleted the lines containing the following libraries in the file: torch, torchvision, torch-geometric,  torch-scatter, torch-sparse, numpy
pip install -r requirements.txt

pip install numba==0.55.1

conda install -c conda-forge hdbscan==0.8.27

conda install numpy-base==1.19.2

pip install joblib==1.1.0
```

### Example 2 of installation
Specs local computer: Ubuntu 22.04.1, 64-bit, CUDA version 11.3
```bash
conda create -n torchpoint3denv python=3.8
conda activate torchpoint3denv
conda install -c conda-forge gcc==9.4.0
conda install pytorch=1.9.0 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
conda install numpy==1.19.5
mamba install libopenblas openblas
find ${CONDA_PREFIX}/include -name "cblas.h"
export CXX=g++
export MAX_JOBS=2;
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas

#THE STEPS START FROM HERE ARE EXACT THE SAME AS EXAMPLE 1 
#CHECK IF TORCH AND MINKOWSKI ENGINE WORK AS EXPECTED: 
#...

```

Based on our experience, we would suggest build most of the packages from the source for a larger chance of succesful installation. Good luck to your installation!

# Data introduction

## NPM3D dataset with instance labels
Link for dataset download:
https://polybox.ethz.ch/index.php/s/3lRK77aKdOVQUVB

### Semantic labels
1: "ground",
2: "buildings",
3: "poles",
4: "bollards",
5: "trash cans",
6: "barriers",
7: "pedestrians",
8: "cars",
9: "natural"

### Data folder structure
```bash
├─ conf                    # All configurations for training and evaluation leave there
├─ forward_scripts         # Script that runs a forward pass on possibly non annotated data
├─ outputs                 # All outputs from your runs
├─ scripts                 # Some scripts to help manage the project
├─ torch_points3d
├─ data                    # DATA FOLDER
    └─ npm3dfused
        └─ raw
            ├─ *_train.ply          # All train files
            ├─ *_val.ply            # All val files
            └─ *_test.ply           # All test files
├─ train.py                # Main script to launch a training
└─ eval.py                 # Eval script
```

## FOR-instance dataset
Link for dataset download:

#TO BE ADDED#

### Different forest regions/subgroups of trees
|<h3> Forest region </h3>|<h3> Number of files <h3>|<h3> Approx. largest tree heights </h3>|
| :--------------------: | :---------------------: | :-------------------------------: |
| <h3> CULS </h3> | 3 | 25|
| <h3> NIBIO </h3> | 20 | 30|
| <h3> NIBIO2 </h3> | 50 | 25|
| <h3> RMIT </h3> | 2 | 15|
| <h3> SCION </h3> | 5 | 30|
| <h3> TUWIEN </h3> | 2 | 35|

### Train - validation - test split
Train - test split is given by NIBIO: 56 train files, 26 test files. We decided on choosing 25% of the train files randomly but fixed as validation set -> 42 train files, 14 val files, 26 test files. 

### Data folder structure
```bash
├─ conf                    # All configurations for training and evaluation leave there
├─ forward_scripts         # Script that runs a forward pass on possibly non annotated data
├─ outputs                 # All outputs from your runs
├─ scripts                 # Some scripts to help manage the project
├─ torch_points3d
├─ data                    # DATA FOLDER
    └─ treeinsfused
        └─ raw
            ├─ CULS
                ├─ *_train.ply          # All train files
                ├─ *_val.ply            # All val files
                └─ *_test.ply           # All test files
            ├─ NIBIO
                ├─ *_train.ply          # SIMILAR AS CULS FOLDER
                ├─ *_val.ply            
                └─ *_test.ply           
            ├─ NIBIO2
                ├─ *.ply                # SIMILAR AS CULS FOLDER          
            ├─ RMIT
                ├─ *.ply                # SIMILAR AS CULS FOLDER
            ├─ SCION
                ├─ *.ply                # SIMILAR AS CULS FOLDER
            └─ TUWIEN
                ├─ *.ply                # SIMILAR AS CULS FOLDER
├─ train.py                # Main script to launch a training
└─ eval.py                 # Eval script
```

# Getting started with code

## How to set different parameters

|<h3> Parameter </h3>|<h3> Value <h3>|<h3> Where to find/How to set in code </h3>|
| :--------------------: | :---------------------: | :-------------------------------: |
| Choose different settings in Table 2 in the original paper| Setting I-V| Setting I: models=panoptic/area4_ablation_19, Setting II: models=panoptic/area4_ablation_14, Setting III: models=panoptic/area4_ablation_15, Setting IV: models=panoptic/area4_ablation_3heads_5 Setting V: models=panoptic/area4_ablation_3heads_6 |
| Number of training iterations | 150 epochs | conf/training/#NameOfYourChosenConfigFile#.yaml, line 3: epochs|
| Voxel size/subsampling size | 0.2 (m) | conf/data/panoptic/#NameOfYourChosenConfigFile#.yaml, line 11: first_subsampling |
| Radius of sampling cylinder | 8 (m) | conf/data/panoptic/#NameOfYourChosenConfigFile#.yaml, line 12: radius |
| The folder name of your output files | #YourOutputFolderName# | job_name=#YourOutputFolderName# |

1. Create wandb account and specify your own wandb account name in conf/training/*.yaml. Have a look at all needed configurations of your current run in conf/data/panoptic/*.yaml, conf/models/panoptic/*.yaml and conf/training/*.yaml. Perform training by running:

```bash
#An example for NPM3D dataset
# Run Setting IV for test area 1, radius=16m, voxel side length=0.12m, training for 200 epoches.
python train.py task=panoptic data=panoptic/npm3d-sparseconv_grid_012_R_16_cylinder_area1 models=panoptic/area4_ablation_3heads_5 model_name=PointGroup-PAPER training=7_area1 job_name=A1_S7

#An example for FOR-instance dataset
# Run Setting IV, radius=8m, voxel side length=0.2m, training for 150 epoches.
python train.py task=panoptic data=panoptic/treeins models=panoptic/area4_ablation_3heads model_name=PointGroup-PAPER training=treeins job_name=treeins_my_first_run
```

2. Look at "TO ADAPT" comments in conf/eval.yaml and change accordingly. Perform evaluation by running:
```bash
python eval.py
```

3. Look at "TO ADAPT" comments in evaluation_stats.py and change accordingly, then run:
```bash
python evaluation_stats.py
```

# Citing
If you find our work useful, please do not hesitate to cite it:

```
@inproceedings{
  Xiang2023,
  title={Toward Accurate Instance Segmentation in Large-scale LiDAR Point Clouds},
  author={Binbin Xiang and Torben Peters and Theodora Kontogianni and Frawa Vetterli1 and Stefano Puliti and Rasmus Astrup and Konrad Schindler},
  booktitle={2023 The ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
  year={2023},
  url = {\url{To be added}}
}
```