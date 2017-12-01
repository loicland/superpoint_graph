Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
=========

This is the official PyTorch implementation of our paper *Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs* <http://arxiv.org/abs/1711.09869>.

<img src="http://imagine.enpc.fr/~simonovm/largescale/teaser.jpg" width="900">


## Code structure
* `./?????` - Partition code (geometric partitioning and superpoint graph construction)
* `./learning/*` - Learning code (superpoint embedding and contextual segmentation).


## Requirements

1. Install [PyTorch](https://pytorch.org) and then [torchnet](https://github.com/pytorch/tnt) with `pip install git+https://github.com/pytorch/tnt.git@master`.

2. Install additional Python packages: `pip install future python-igraph tqdm transforms3d pynvrtc cupy h5py sklearn`.

The code was tested on Ubuntu 14.04 with Python 3.6 and PyTorch 0.2.



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



### S3DIS

To train on the all 6 folds, run

```
for FOLD in 1 2 3 4 5 6; do \
CUDA_VISIBLE_DEVICES=0 python learning/main.py --dataset s3dis --cvfold $FOLD --epochs 350 --lr_steps '[275,320]' \
--test_nth_epoch 50 --model_config 'gru_10,f_13' --ptn_nfeat_stn 14 --nworkers 2 --odir "results/s3dis/best/cv${FOLD}"; \
done
```

The trained networks can be downloaded [here](http://imagine.enpc.fr/~simonovm/largescale/models_s3dis.zip), unzipped and loaded with `--resume` argument.

Note that you can use `--S3DIS_PATH` argument to set path to the pre-processed dataset.














