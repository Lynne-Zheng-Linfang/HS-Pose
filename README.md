# HS-Pose (CVPR 2023)
Pytorch implementation of HS-Pose: Hybrid Scope Feature Extraction for Category-level Object Pose Estimation.
([Paper](https://arxiv.org/abs/2303.15743), [Project](https://lynne-zheng-linfang.github.io/hspose.github.io/))


![teaser](pic/teaser.png)
<p align="center">
    Illstraction of the hybrid feature extraction.
</p>

![pipeline](pic/pipeline.png)
<p align="center">
    The overall framework.
</p>


## UPDATE!


## Required environment
- Ubuntu 18.04
- Python 3.8 
- Pytorch 1.10.1
- CUDA 11.2
- 1 * RTX 3090
 

## Virtual environment
```shell
cd HS-Pose
virtualenv HS-Pose-env -p /usr/bin/python3.8
```
Then, copy past the following lines to the end of `./HS-Pose-env/bin/activate` file:
```shell
CUDAVER=cuda-11.2
export PATH=/usr/local/$CUDAVER/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/$CUDAVER/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/$CUDAVER/lib64:$LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/$CUDAVER
export CUDA_ROOT=/usr/local/$CUDAVER
export CUDA_HOME=/usr/local/$CUDAVER
```
Then, use `source` to activate the virtualenv:
```shell
source HS-Pose-env/bin/activate
```


## Installing
- Install [CUDA-11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=2004&target_type=deblocal) 

- Install basic packages:
```shell
chmod +x env_setup.sh
./env_setup.sh
```
<!-- - Install [Detectron2](https://github.com/facebookresearch/detectron2). -->

## Data Preparation
To generate your own dataset, use the data preprocess code provided in this [git](https://github.com/mentian/object-deformnet/blob/master/preprocess/pose_data.py). Download the detection results in this [git](https://github.com/Gorilla-Lab-SCUT/DualPoseNet). Change the `dataset_dir` and `detection_dir` to your own path.

Since the handle visibility labels are not provided in the original NOCS REAL275 train set, please put the handle visibility file `./mug_handle.pkl` under `YOUR_NOCS_DIR/Real/train/` folder. The `mug_handle.pkl` is mannually labeled and originally provided by the [GPV-Pose](https://github.com/lolrudy/GPV_Pose).


## Trained model
### REAL275
Download the trained model from this [google link](https://drive.google.com/file/d/1TszIS5ebECVpLyEbukOhb7QhVIwPeTIM/view?usp=sharing) or [baidu link](https://pan.baidu.com/s/1Y8Gb0azh7lWt8XEgfNY_cw) (code: w8pw). After downloading it, please extracted it and then put the `HS-Pose_weights` folder into the `output/models/` folder. 

Run the following command to check the results for REAL275 dataset:
```shell
python -m evaluation.evaluate  --model_save output/models/HS-Pose_weights/eval_result --resume 1 --resume_model ./output/models/HS-Pose_weights/model.pth --eval_seed 1677483078
```
### CAMERA25
Download the trained model from this [google link](https://drive.google.com/file/d/1_Dcy-VXcMABihusLDVV_axibFBW-JjJF/view?usp=sharing) or [baidu link](https://pan.baidu.com/s/1QNLPxJn86Gk-mxVHsjTlsQ) (code: 9et7). After downloading it, please extracted it and then put the `HS-Pose_CAMERA25_weights` folder into the `output/models/` folder. 

Run the following command to check the results for CAMERA25 dataset:
```shell
python -m evaluation.evaluate  --model_save output/models/HS-Pose_CAMERA25_weights/eval_result --resume 1 --resume_model ./output/models/HS-Pose_CAMERA25_weights/model.pth --eval_seed 1678917637 --dataset CAMERA
```

## Training
Please note, some details are changed from the original paper for more efficient training. 

Specify the dataset directory and run the following command.
```shell
python -m engine.train --dataset_dir YOUR_DATA_DIR --model_save SAVE_DIR
```

Detailed configurations are in `config/config.py`.

## Evaluation
```shell
python -m evaluation.evaluate --dataset_dir YOUR_DATA_DIR --detection_dir DETECTION_DIR --resume 1 --resume_model MODEL_PATH --model_save SAVE_DIR
```

## Example Code
You can run the following training and testing commands to get the results similar to the below table.
```shell
python -m engine.train --model_save output/models/HS-Pose/ --num_workers 20 --batch_size 16 --train_steps 1500 --seed 1677330429 --dataset_dir YOUR_DATA_DIR --detection_dir DETECTION_DIR
python -m evaluation.evaluate  --model_save output/models/HS-Pose/model_149 --resume 1 --resume_model ./output/models/HS-Pose/model_149.pth --eval_seed 1677483078 --dataset_dir YOUR_DATA_DIR --detection_dir DETECTION_DIR
```
|Metrics| IoU25 | IoU50 | IoU75 | 5d2cm | 5d5cm | 10d2cm| 10d5cm| 10d10cm|  5d   | 2cm   |
|:------|:------|:------|:------|:------|:------|:------|:------|:-------|:------|:------|
|Scores | 84.3  | 82.8  | 75.3  |  46.2 |  56.1 | 68.9  | 84.1  | 85.2   | 59.1  | 77.8  |




## Citation
Cite us if you found this work useful.
```
@misc{zheng2023hspose,
      title={HS-Pose: Hybrid Scope Feature Extraction for Category-level Object Pose Estimation}, 
      author={Linfang Zheng and Chen Wang and Yinghan Sun and Esha Dasgupta and Hua Chen and Ales Leonardis and Wei Zhang and Hyung Jin Chang},
      year={2023},
      eprint={2303.15743},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## Acknowledgment
Our implementation leverages the code from [3dgcn](https://github.com/j1a0m0e4sNTU/3dgcn), [FS-Net](https://github.com/DC1991/FS_Net),
[DualPoseNet](https://github.com/Gorilla-Lab-SCUT/DualPoseNet), [SPD](https://github.com/mentian/object-deformnet), [GPV-Pose](https://github.com/lolrudy/GPV_Pose).