# Semantic3D

Download all point clouds and labels from [Semantic3D Dataset](http://www.semantic3d.net/) and place extracted training files to `$SEMA3D_DIR/data/train`, reduced test files into `$SEMA3D_DIR/data/test_reduced`, and full test files into `$SEMA3D_DIR/data/test_full`, where `$SEMA3D_DIR` is set to dataset directory. The label files of the training files must be put in the same directory than the .txt files.

## Handcrafted Partition

To compute the partition with handcrafted features run:
```
python partition/partition.py --dataset sema3d --ROOT_PATH $SEMA3D_DIR --voxel_width 0.05 --reg_strength 0.8 --ver_batch 5000000
```
It is recommended that you have at least 24GB of RAM to run this code. Otherwise, increase the ```voxel_width``` parameter to increase pruning.

Then, reorganize point clouds into superpoints by:
```
python learning/sema3d_dataset.py --SEMA3D_PATH $SEMA3D_DIR
```

To train on the whole publicly available data and test on the reduced test set, run:
```
CUDA_VISIBLE_DEVICES=0 python learning/main.py --dataset sema3d --SEMA3D_PATH $SEMA3D_DIR --db_test_name testred --db_train_name trainval \
--epochs 500 --lr_steps '[350, 400, 450]' --test_nth_epoch 100 --model_config 'gru_10,f_8' --ptn_nfeat_stn 11 \
--nworkers 2 --pc_attrib xyzrgbelpsv --odir "results/sema3d/trainval_best"
```
The trained network can be downloaded [here](http://imagine.enpc.fr/~simonovm/largescale/model_sema3d_trainval.pth.tar) and loaded with `--resume` argument. Rename the file ```model.pth.tar``` (do not try to unzip it!) and place it in the directory ```results/sema3d/trainval_best```.

To test this network on the full test set, run:
```
CUDA_VISIBLE_DEVICES=0 python learning/main.py --dataset sema3d --SEMA3D_PATH $SEMA3D_DIR --db_test_name testfull --db_train_name trainval \
--epochs -1 --lr_steps '[350, 400, 450]' --test_nth_epoch 100 --model_config 'gru_10,f_8' --ptn_nfeat_stn 11 \
--nworkers 2 --pc_attrib xyzrgbelpsv --odir "results/sema3d/trainval_best" --resume RESUME
```
We validated our configuration on a custom split of 11 and 4 clouds. The network is trained as such:
```
CUDA_VISIBLE_DEVICES=0 python learning/main.py --dataset sema3d --SEMA3D_PATH $SEMA3D_DIR --epochs 450 --lr_steps '[350, 400]' --test_nth_epoch 100 \
--model_config 'gru_10,f_8' --pc_attrib xyzrgbelpsv --ptn_nfeat_stn 11 --nworkers 2 --odir "results/sema3d/best"
```

#### Learned Partition

Not yet available.

#### Visualization

To upsample the prediction to the unpruned data and write the .labels files for the reduced test set, run (quite slow):
```
python partition/write_Semantic3d.py --SEMA3D_PATH $SEMA3D_DIR --odir "results/sema3d/trainval_best" --db_test_name testred
```

To visualize the results and intermediary steps (on the subsampled graph), use the visualize function in partition. For example:
```
python partition/visualize.py --dataset sema3d --ROOT_PATH $SEMA3D_DIR --res_file 'results/sema3d/trainval_best/prediction_testred' --file_path 'test_reduced/MarketplaceFeldkirch_Station4' --output_type ifprs
```
avoid ```--upsample 1``` as it can can take a very long time on the largest clouds.
