# Masked Autoencoders for Self-Supervised Learning on Automotive Point Clouds - Zenseact Opensource Dataset (ZOD)

![Alt text](/main_voxel/asset/image.png "ZOD single frame")

## Usage

**PyTorch >= 1.9 is recommended for a better support of the checkpoint technique.**


The implementation builds upon code from [SST](https://github.com/TuSimple/SST), which in turn is based on [MMDetection3D](https://github.com/open-mmlab/mmdetection3d). Please refer to their [getting_started](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/getting_started.md) for getting the environment up and running.

### Environment 

####  Create an environment based on a Dockerfile. 

build the Docker image using the "docker build" command.

The docker build command with the `-f` and `-t` options is used to build a Docker image from a Dockerfile, and the `$DOCKER_ROOT` argument specifies the build context.

```
docker build -f Dockerfile -t $DOCKER_NAME $DOCKER_ROOT
```
Once the image is built, you can create a new container based on the image using the `docker run` command. 

##### Build a Dockerfile using a bash file

you can make the `build.sh` file executable and then execute the `build.sh` file by running the following command:

```
chmod +x build.sh

./build.sh
```

### Data Preparation

The folder sructure should be organized as follows before preparation:

```bash
main_voxel
├── mmdet3d
├── tools
├── configs
├── data
│   ├── zod
│   │   ├── single_frames
│   │   ├── trainval_frames_mini.json
│   │   ├── trainval_frames_full.json
```

We typically need to organize the useful data information with a .pkl or .json file in a specific style. To prepare these files, run the following commands:

```bash
python tools/create_data.py zod --root-path ./data/zod --out-dir ./data/zod --extra-tag mini
```

### Training models
The training procedure is the same as the one in SST. Please refer to `./tools/train.py` or `./tools/dist_train.sh` for details.

### Pre-training models

In `./configs/sst_masked/`, we provide the configs used for pre-training. 

The config with suffix `intensity.py` use intensity information. `remove-close` refers to removal of points hitting the ego-vehicle. `cpf` refers to using the three pre-training tasks (Chamfer, #points and "fake"/empty voxels). `200e` refers to the number of epochs used for pre-training.

For pre-training with a subset of the pre-training tasks, use variations of `cpf`, e.g. `cf` refers to using Chamfer and fake voxels, `pf` refers to using #points and fake voxels, etc.


### Fine-tuning models

After pre-training, we can use the pre-trained checkpoints to initialize the 3D OD model. Again, training is started with `tools/train.py` or `tools/dist_train.


 However, to load the pre-trained weights, we need to use the `--cfg-options` option with `load_from`. For instance, `tools/dist_train.sh $CONFIG $GPUS --cfg-options load_from=$PATH_TO_PRETRAINED_CHECKPOINT`. Configs for fine-tuning can be found in `./configs/sst_refactor`.


## Acknowledgments
This project is based on the following codebases.  
The official implementation of the WACV 2023 PVL workshop paper [Masked Autoencoders for Self-Supervised Learning on Automotive Point Clouds](https://arxiv.org/abs/2207.00531).

* [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
* [LiDAR-RCNN](https://github.com/TuSimple/LiDAR_RCNN)
* [SST](https://github.com/TuSimple/SST)
