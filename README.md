Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
=========

This is the official PyTorch implementation of our paper *Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs* <http://arxiv.org/abs/1711.09869>.

<img src="http://imagine.enpc.fr/~simonovm/largescale/teaser.jpg" width="900">


## Code structure
* `./partition/*` - Partition code (geometric partitioning and superpoint graph construction)
* `./learning/*` - Learning code (superpoint embedding and contextual segmentation).


## Requirements

1. Install [PyTorch](https://pytorch.org), [visdom](https://anaconda.org/conda-forge/visdom) with ```conda install -c conda-forge visdom``` and finally [torchnet](https://github.com/pytorch/tnt) with `pip install git+https://github.com/pytorch/tnt.git@master`.

2. Install additional Python packages: `pip install future python-igraph tqdm transforms3d pynvrtc cupy h5py sklearn plyfile scipy`.

3. Update Boost to version 1.63.0 or newer, in Conda: `conda install -c anaconda boost`

4. Compile the ```libply_c``` and ```libcp``` libraries in /partition/
```
cd ply_c
cmake .
make
cd ..
cd cut-pursuit
cmake .
make
```
The code was tested on Ubuntu 14.04 with Python 3.6 and PyTorch 0.2.

## Partition

### S3DIS

To compute the partition run

```python partition_S3DIS.py```

### Semantic3D

To compute the partition run

```python partition_Semantic3D.py```

It is recommended that you have at least 24GB of RAM to run this code. Otherwise, either use swap memory of increase the ```voxel_width``` parameter to increase pruning.

## Learning

### Semantic3D

To train on the whole publicly available data and test on the reduced test set, run
```
CUDA_VISIBLE_DEVICES=0 python learning/main.py --dataset sema3d --db_test_name testred --db_train_name trainval \
--epochs 500 --lr_steps '[350, 400, 450]' --test_nth_epoch 100 --model_config 'gru_10,f_8' --ptn_nfeat_stn 11 \
--nworkers 2 --odir "results/sema3d/trainval_best"
```
The trained network can be downloaded [here](http://imagine.enpc.fr/~simonovm/largescale/model_sema3d_trainval.pth.tar) and loaded with `--resume` argument.


To test this network on the full test set, run
```
CUDA_VISIBLE_DEVICES=0 python learning/main.py --dataset sema3d --db_test_name testfull --db_train_name trainval \
--epochs -1 --lr_steps '[350, 400, 450]' --test_nth_epoch 100 --model_config 'gru_10,f_8' --ptn_nfeat_stn 11 \
--nworkers 2 --odir "results/sema3d/trainval_best" --resume RESUME
```

We validated our configuration on a custom split of 11 and 4 clouds. The network is trained as such:
```
CUDA_VISIBLE_DEVICES=0 python learning/main.py --dataset sema3d --epochs 450 --lr_steps '[350, 400]' --test_nth_epoch 100 \
--model_config 'gru_10,f_8' --ptn_nfeat_stn 11 --nworkers 2 --odir "results/sema3d/best"
```

Note that you can use `--SEMA3D_PATH` argument to set path to the pre-processed dataset.

# Licence

SPGraph is under a dual GPL3.0 / commercial license. If you want to use SPGraph for commercial, non-GPL use, contact us about commercial licensing, which will be determined on a case-by-case basis.
