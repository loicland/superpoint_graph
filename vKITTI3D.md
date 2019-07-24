# vKITTI3D

Download all point clouds and labels from [the vKITTI3D Dataset](https://github.com/VisualComputingInstitute/vkitti3D-dataset) and place extracted training files to `$VKITTI3D_DIR/data/`.

## Handcrafted Partition

Not available for this dataset, the sparsity of the acquisition renders the handcrafted geometric features useless.

## Learned Partition

For the learned partition, run:
```
python supervized_partition/graph_processing.py --ROOT_PATH $VKITTI3D_DIR --dataset vkitti --voxel_width 0.05 --use_voronoi 1

for FOLD in 1 2 3 4 5 6; do
    python ./supervized_partition/supervized_partition.py --ROOT_PATH $VKITTI3D_DIR --dataset vkitti \
    --epochs 50 --test_nth_epoch 10  --cvfold $FOLD --reg_strength 0.5 --spatial_emb 0.02 --batch_size 15 \
    --global_feat exyrgb --CP_cutoff 10 --odir results_part/vkitti/best; \
 done;   
```
 or use our [trained weights](http://recherche.ign.fr/llandrieu/SPG/vkitti/results_part/pretrained.zip) and the `--resume RESUME` argument:
 
 ```
python supervized_partition/graph_processing.py --ROOT_PATH $VKITTI3D_DIR --dataset vkitti --voxel_width 0.05 --use_voronoi 1

for FOLD in 1 2 3 4 5 6; do
    python ./supervized_partition/supervized_partition.py --ROOT_PATH $VKITTI3D_DIR --dataset vkitti \
    --epochs -1 --test_nth_epoch 10 --cvfold $FOLD --reg_strength 0.5 --spatial_emb 0.02 --batch_size 15\
    --global_feat exyrgb --CP_cutoff 10 --odir results_partition/vkitti/pretrained --resume RESUME; \
 done;   
 ```
 
 To evaluate the quality of the partition, run:
 ```
 python supervized_partition/evaluate_partition.py --dataset vkitti --cvfold 123456 --folder best
```
### Training

Then, reorganize point clouds into superpoints by:
```
python learning/vkitti_dataset.py --VKITTI_PATH $VKITTI3D_DIR
```

To train from scratch, run:
```
for FOLD in 1 2 3 4 5 6; do \ 
CUDA_VISIBLE_DEVICES=0 python ./learning/main.py --dataset vkitti --VKITTI_PATH $VKITTI3D_DIR --cvfold $FOLD --epochs 100 \ --lr_steps "[40, 50, 60, 70, 80]" --test_nth_epoch 10 --model_config gru_10_1_1_1_0,f_13 --pc_attribs xyzXYZrgb \ 
--ptn_nfeat_stn 9 --batch_size 4 --ptn_minpts 15 --spg_augm_order 3 --spg_augm_hardcutoff 256 \
--ptn_widths "[[64,64,128], [64,32,32]]" --ptn_widths_stn "[[32,64], [32,16]]" --loss_weights sqrt \
--use_val_set 1 --odir results/vkitti/best/cv$FOLD; \
done;\
```

or use our [trained weights](http://recherche.ign.fr/llandrieu/SPG/vkitti/results/pretrained.zip) with the `--resume RESUME` argument:
```
for FOLD in 1 2 3 4 5 6; do \ 
CUDA_VISIBLE_DEVICES=0 python ./learning/main.py --dataset vkitti --VKITTI_PATH $VKITTI3D_DIR --cvfold $FOLD --epochs 100 \ --lr_steps "[40, 50, 60, 70, 80]" --test_nth_epoch 10 --model_config gru_10_1_1_1_0,f_13 --pc_attribs xyzXYZrgb \ 
--ptn_nfeat_stn 9 --batch_size 4 --ptn_minpts 15 --spg_augm_order 3 --spg_augm_hardcutoff 256 \
--ptn_widths "[[64,64,128], [64,32,32]]" --ptn_widths_stn "[[32,64], [32,16]]" --loss_weights sqrt \
--use_val_set 1 --odir results/vkitti/pretrained/cv$FOLD --resume RESUME; \
done;\
```

Estimate the quality of the semantic segmentation with:
```
python learning/evaluate.py --dataset vkitti --odir results/vkitti/best --cvfold 123456
```
#### Visualization

To visualize the results and intermediary steps (on the subsampled graph), use the visualize function in partition. For example:
```
python partition/visualize.py --dataset vkitti --ROOT_PATH $VKITTI3D_DIR --res_file 'results/vkitti/cv1/predictions_test' --file_path '01/0001_00000' --output_type ifprs
```
