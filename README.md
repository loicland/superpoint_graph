
# Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs


This is the official PyTorch implementation of our paper *Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs* <http://arxiv.org/abs/1711.09869>.

<img src="http://imagine.enpc.fr/~simonovm/largescale/teaser.jpg" width="900">


## Code structure
* `./partition/*` - Partition code (geometric partitioning and superpoint graph construction)
* `./learning/*` - Learning code (superpoint embedding and contextual segmentation).


## Requirements

1. Install [PyTorch](https://pytorch.org) and [torchnet](https://github.com/pytorch/tnt) with `pip install git+https://github.com/pytorch/tnt.git@master`. Pytorch 0.4 is not tested and might cause errors.

2. Install additional Python packages: `pip install future python-igraph tqdm transforms3d pynvrtc fastrlock cupy h5py sklearn plyfile scipy`.

3. Install Boost (1.63.0 or newer) and Eigen3, in Conda: `conda install -c anaconda boost; conda install -c omnia eigen3; conda install eigen; conda install -c r libiconv`.

4. Make sure that cut pursuit was downloaded. Otherwise, clone [this repository](https://github.com/loicland/cut-pursuit) in `/partition`

5. Compile the ```libply_c``` and ```libcp``` libraries:
```
cd partition/ply_c
cmake . -DPYTHON_LIBRARY=$CONDAENV/lib/libpython3.6m.so -DPYTHON_INCLUDE_DIR=$CONDAENV/include/python3.6m -DBOOST_INCLUDEDIR=$CONDAENV/include -DEIGEN3_INCLUDE_DIR=$CONDAENV/include/eigen3
make
cd ..
cd cut-pursuit/src
cmake . -DPYTHON_LIBRARY=$CONDAENV/lib/libpython3.6m.so -DPYTHON_INCLUDE_DIR=$CONDAENV/include/python3.6m -DBOOST_INCLUDEDIR=$CONDAENV/include -DEIGEN3_INCLUDE_DIR=$CONDAENV/include/eigen3
make
```
where `$CONDAENV` is the path to your conda environment. The code was tested on Ubuntu 14.04 with Python 3.6 and PyTorch 0.2 and 0.3. The newer 0.4 is not tested yet.

### Troubleshooting

Common sources of error and how to fix them:
- $CONDA_ENV is not defined : define it or replace $CONDA_ENV by the absolute path of your environment (find it with ```locate anaconda```)
- anaconda uses a different version of python than 3.6m : adapt it in the command. Find which version of python conda is using with ```locate anaconda3/lib/libpython```
- you are using boost 1.62 or older: update it
- cut pursuit did not download: manually clone it in the ```partition``` folder.

## S3DIS

Download [S3DIS Dataset](http://buildingparser.stanford.edu/dataset.html) and extract `Stanford3dDataset_v1.2_Aligned_Version.zip` to `$S3DIR_DIR/data`, where `$S3DIR_DIR` is set to dataset directory.

To fix some issues with the dataset as reported in issue [#29](https://github.com/loicland/superpoint_graph/issues/29), apply path `S3DIS_fix.diff` with:
```cp S3DIS_fix.diff $S3DIR_DIR/data; cd $S3DIR_DIR/data; git apply S3DIS_fix.diff; rm S3DIS_fix.diff; cd -```

### Partition

To compute the partition run

```python partition/partition.py --dataset s3dis --ROOT_PATH $S3DIR_DIR --voxel_width 0.03 --reg_strength 0.03```

### Training

First, reorganize point clouds into superpoints by:

```python learning/s3dis_dataset.py --S3DIS_PATH $S3DIR_DIR```

To train on the all 6 folds, run
```
for FOLD in 1 2 3 4 5 6; do \
CUDA_VISIBLE_DEVICES=0 python learning/main.py --dataset s3dis --S3DIS_PATH $S3DIR_DIR --cvfold $FOLD --epochs 350 --lr_steps '[275,320]' \
--test_nth_epoch 50 --model_config 'gru_10_0,f_13' --ptn_nfeat_stn 14 --nworkers 2 --odir "results/s3dis/best/cv${FOLD}"; \
done
```
The trained networks can be downloaded [here](http://imagine.enpc.fr/~simonovm/largescale/models_s3dis.zip), unzipped and loaded with `--resume` argument.

To test this network on the full test set, run
```
for FOLD in 1 2 3 4 5 6; do \
CUDA_VISIBLE_DEVICES=0 python learning/main.py --dataset s3dis --S3DIS_PATH $S3DIR_DIR --cvfold $FOLD --epochs -1 --lr_steps '[275,320]' \
--test_nth_epoch 50 --model_config 'gru_10_0,f_13' --ptn_nfeat_stn 14 --nworkers 2 --odir "results/s3dis/best/cv${FOLD}" --resume RESUME; \
done
```

To evaluate quantitavily on the full set on a trained model type: 
```python learning/evaluate_s3dis.py --odir results/s3dis/best --cvfold 123456``` 

To visualize the results and all intermediary steps, use the visualize function in partition. For example:
```
python partition/visualize.py --dataset s3dis --ROOT_PATH $S3DIR_DIR --res_file 'models/cv1/predictions_val' --file_path 'Area_1/conferenceRoom_1' --output_type igfpres
```

```output_type``` defined as such:
- ```'i'``` = input rgb point cloud
- ```'g'``` = ground truth (if available), with the predefined class to color mapping
- ```'f'``` = geometric feature with color code: red = linearity, green = planarity, blue = verticality
- ```'p'``` = partition, with a random color for each superpoint
- ```'r'``` = result cloud, with the predefined class to color mapping
- ```'e'``` = error cloud, with green/red hue for correct/faulty prediction 
- ```'s'``` = superedge structure of the superpoint (toggle wireframe on meshlab to view it)

Add option ```--upsample 1``` if you want the prediction file to be on the original, unpruned data.

## Semantic3D

Download all point clouds and labels from [Semantic3D Dataset](http://www.semantic3d.net/) and place extracted training files to `$SEMA3D_DIR/data/train`, reduced test files into `$SEMA3D_DIR/data/test_reduced`, and full test files into `$SEMA3D_DIR/data/test_full`, where `$SEMA3D_DIR` is set to dataset directory. The label files of the training files must be put in the same directory than the .txt files.

### Partition

To compute the partition run

```python partition/partition.py --dataset sema3d --ROOT_PATH $SEMA3D_DIR --voxel_width 0.05 --reg_strength 0.8 --ver_batch 5000000```

It is recommended that you have at least 24GB of RAM to run this code. Otherwise, increase the ```voxel_width``` parameter to increase pruning.

### Training

First, reorganize point clouds into superpoints by:

```python learning/sema3d_dataset.py --SEMA3D_PATH $SEMA3D_DIR```

To train on the whole publicly available data and test on the reduced test set, run
```
CUDA_VISIBLE_DEVICES=0 python learning/main.py --dataset sema3d --SEMA3D_PATH $SEMA3D_DIR --db_test_name testred --db_train_name trainval \
--epochs 500 --lr_steps '[350, 400, 450]' --test_nth_epoch 100 --model_config 'gru_10,f_8' --ptn_nfeat_stn 11 \
--nworkers 2 --odir "results/sema3d/trainval_best"
```
The trained network can be downloaded [here](http://imagine.enpc.fr/~simonovm/largescale/model_sema3d_trainval.pth.tar) and loaded with `--resume` argument. Rename the file ```model.pth.tar``` (do not try to unzip it!) and place it in the directory ```results/sema3d/trainval_best```.

To test this network on the full test set, run
```
CUDA_VISIBLE_DEVICES=0 python learning/main.py --dataset sema3d --SEMA3D_PATH $SEMA3D_DIR --db_test_name testfull --db_train_name trainval \
--epochs -1 --lr_steps '[350, 400, 450]' --test_nth_epoch 100 --model_config 'gru_10,f_8' --ptn_nfeat_stn 11 \
--nworkers 2 --odir "results/sema3d/trainval_best" --resume RESUME
```

We validated our configuration on a custom split of 11 and 4 clouds. The network is trained as such:
```
CUDA_VISIBLE_DEVICES=0 python learning/main.py --dataset sema3d --SEMA3D_PATH $SEMA3D_DIR --epochs 450 --lr_steps '[350, 400]' --test_nth_epoch 100 \
--model_config 'gru_10,f_8' --ptn_nfeat_stn 11 --nworkers 2 --odir "results/sema3d/best"
```

To upsample the prediction to the unpruned data and write the .labels files for the reduced test set, run:

```python partition/write_Semantic3d.py --SEMA3D_PATH $SEMA3D_DIR --odir "results/sema3d/trainval_best" --db_test_name testred```

To visualize the results and intermediary steps (on the subsampled graph), use the visualize function in partition. For example:
```
python partition/visualize.py --dataset sema3d --ROOT_PATH $SEMA3D_DIR --res_file 'results/sema3d/trainval_best/prediction_testred' --file_path 'test_reduced/MarketplaceFeldkirch_Station4' --output_type ifprs
```

avoid ```--upsample 1``` as it can can take a very long time on the largest clouds.

# Other data sets

You can apply SPG on your own data set with minimal changes:
- adapt references to ```custom_dataset``` in ```/partition/partition.py```
- you will need to create the function ```read_custom_format``` in ```/partition/provider.py``` which outputs xyz and rgb values, as well as semantic labels if available (already implemented for ply and las files)
- adapt the template function ```/learning/custom_dataset.py``` to your achitecture and design choices
- adapt references to ```custom_dataset``` in ```/learning/main.py```
- add your data set colormap to ```get_color_from_label``` in ```/partition/provider.py```
- adapt line 212 of `learning/spg.py` to reflect the missing or extra point features
- change ```--model_config``` to ```gru_10,f_K``` with ```K``` as the number of classes in your dataset, or ```gru_10_0,f_K``` to use matrix edge filters instead of vectors (only use matrices when your data set is quite large, and with many different point clouds, like S3DIS).

# Datasets without RGB
If your data does not have RGB values you can easily use SPG. You will need to follow the instructions in ```partition/partition.ply``` regarding the pruning.
You will need to adapt the ```/learning/custom_dataset.py``` file so that it does not refer ro RGB values.
You should absolutely not use a model pretrained on values with RGB. instead, retrain a model from scratch using the ```--pc_attribs xyzelpsv``` option to remove RGB from the shape embedding input and setting the ```--pc_attribs``` option to the correct number of features (7 in case of ```--pc_attribs xyzelpsv```).


## HELIX
Getting the S3DIS dataset (fixed dataset, no need to apply the diff)

https://storage.googleapis.com/helix-dev-jupyterhub-data/point-clouds/S3DIS-dataset/Stanford3dDataset_v1.3_Aligned_Version.tar.gz


Your folder structure should look like (once you have added some custom helix data as well):
```
data
|-- S3DIS
    |-- clouds
    |-- data
        |-- Area_1
        |-- Area_2
        |-- Area_3
        |-- Area_4
        |-- Area_5
        |-- Area_6
    |-- features
    |-- parsed
    |-- superpoint_graphs
|-- helix
    |-- clouds
    |-- data
        |-- test
    |-- features
    |-- parsed
    |-- superpoint_graphs
learning
partition
results
```

### Running inference
For every point cloud, the steps from raw data to segmented cloud are as follow:

1. superpoint_point segmentation, for S3DIS run:
    ```
    python partition/partition.py --dataset s3dis --ROOT_PATH data/S3DIS --voxel_width 0.03 --reg_strength 0.03
    ```
2. parse super_points, for S3DIS:
    ```
    python learning/s3dis_dataset.py --S3DIS_PATH data/S3DIS
    ```
3. predict using a pretrained model, use the notebook provided for that purpose, it contains a visualisation function that writes `ply` files to the `clouds` sub-folder. `ply` files can be viewed with MeshLab for example

### Running on non rgb clouds
Add `.ply` rgb clouds to the `data/helix/data/test` folder and segment them:
```
python partition/partition.py --dataset helix --ROOT_PATH data/helix --voxel_width 0.03 --reg_strength 0.03
```
and then simply follow the steps in the inference notebook. A model trained on non rgb data can be found there:

https://storage.googleapis.com/helix-dev-jupyterhub-data/deep-learning/models/superpoint_graph/norgb/model.pth.tar

It was generated with the following comamnd:
```
learning/main.py --dataset 's3dis' --S3DIS_PATH 'data/S3DIS' --cvfold '1' --epochs '350' --lr_steps '[275,320]' --test_nth_epoch '50' --model_config 'gru_10_0,f_13' --ptn_nfeat_stn '11' --pc_attribs 'xyzelpsvXYZ' --nworkers '8' --odir 'results/s3dis/bw/cv1'
```

### Some more testing data
https://console.cloud.google.com/storage/browser/helix-dev-jupyterhub-data/point-clouds/sample-rooms/?project=helix-dev-195819