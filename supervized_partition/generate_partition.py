import logging
import argparse
import sys
import os
import torch
import glob
import torchnet as tnt
import functools
import tqdm
from multiprocessing import Pool

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(DIR_PATH, '..'))

from supervized_partition import supervized_partition
from supervized_partition import graph_processing
from supervized_partition import losses
from partition import provider
from partition import graphs
from learning import pointnet


def parse_args():
    parser = argparse.ArgumentParser(description='Partition large scale point clouds using cut-pursuit')

    parser.add_argument('--modeldir', help='Folder where the saved model lies', required=True)
    parser.add_argument('--cuda', default=0, type=int, help='Bool, use cuda')
    parser.add_argument('--input_folder',  type=str,
                        help='Folder containing preprocessed point clouds ready for segmentation', required=True)
    parser.add_argument('--output_folder', default="", type=str, help='Folder that will contain the output')
    parser.add_argument('--overwrite', default=1, type=int, help='Overwrite existing partition')
    parser.add_argument('--nworkers', default=5, type=int,
                        help='Num subprocesses to use for generating the SPGs')

    args = parser.parse_args()
    return args


def load_model(model_dir, cuda):
    checkpoint = torch.load(os.path.join(model_dir, supervized_partition.FolderHierachy.MODEL_FILE))
    training_args = checkpoint['args']
    training_args.cuda = cuda  # override cuda
    model = supervized_partition.create_model(training_args)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, training_args


def get_dataloader(input_folder, training_args):
    file_list = glob.glob(os.path.join(input_folder, '*.h5'))
    if not file_list:
        raise ValueError("Empty input folder: %s" % input_folder)
    dataset = tnt.dataset.ListDataset(file_list,
                                      functools.partial(graph_processing.graph_loader, train=False, args=training_args, db_path=""))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         collate_fn=graph_processing.graph_collate)
    return loader


def get_embedder(args):
    if args.learned_embeddings and args.ptn_embedding == 'ptn':
        ptnCloudEmbedder = pointnet.LocalCloudEmbedder(args)
    elif 'geof' in args.ver_value:
        ptnCloudEmbedder = graph_processing.spatialEmbedder(args)
    else:
        raise NameError('Do not know model ' + args.learned_embeddings)
    return ptnCloudEmbedder


def get_num_classes(args):
    # Decide on the dataset
    if args.dataset == 's3dis':
        dbinfo = graph_processing.get_s3dis_info(args)
    elif args.dataset == 'sema3d':
        dbinfo = graph_processing.get_sema3d_info(args)
    elif args.dataset == 'vkitti':
        dbinfo = graph_processing.get_vkitti_info(args)
    else:
        raise NotImplementedError('Unknown dataset ' + args.dataset)
    return dbinfo["classes"]


def process(data_tuple, model, output_folder, training_args, overwrite):
    fname, edg_source, edg_target, is_transition, labels, objects, clouds_data, xyz = data_tuple
    spg_file = os.path.join(output_folder, fname[0])
    logging.info("\nGenerating SPG file %s...", spg_file)
    if os.path.exists(os.path.dirname(spg_file)) and not overwrite:
        logging.info("Already exists, skipping")
        return
    elif not os.path.exists(os.path.dirname(spg_file)):
        os.makedirs(os.path.dirname(spg_file))

    if training_args.cuda:
        is_transition = is_transition.to('cuda', non_blocking=True)
        objects = objects.to('cuda', non_blocking=True)
        clouds, clouds_global, nei = clouds_data
        clouds_data = (clouds.to('cuda', non_blocking=True), clouds_global.to('cuda', non_blocking=True), nei)

    ptnCloudEmbedder = get_embedder(training_args)
    num_classes = get_num_classes(training_args)

    embeddings = ptnCloudEmbedder.run_batch(model, *clouds_data, xyz)

    diff = losses.compute_dist(embeddings, edg_source, edg_target, training_args.dist_type)

    pred_components, pred_in_component = losses.compute_partition(
        training_args, embeddings, edg_source, edg_target, diff, xyz)

    graph_sp = graphs.compute_sp_graph(xyz, 100, pred_in_component, pred_components, labels, num_classes)

    provider.write_spg(spg_file, graph_sp, pred_components, pred_in_component)


def main():
    logging.getLogger().setLevel(logging.INFO)  # set to logging.DEBUG to allow for more prints
    args = parse_args()
    model, training_args = load_model(args.modeldir, args.cuda)
    dataloader = get_dataloader(args.input_folder, training_args)
    workers = max(args.nworkers, 1)

    output_folder = args.output_folder
    if not output_folder:
        # By default assumes that it follows the S3DIS folder structure
        output_folder = os.path.join(args.input_folder, '../..', supervized_partition.FolderHierachy.SPG_FOLDER)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if logging.getLogger().getEffectiveLevel() > logging.DEBUG:
        dataloader = tqdm.tqdm(dataloader, ncols=100)
    with torch.no_grad():
        processing_function = functools.partial(
            process, model=model, output_folder=output_folder, training_args=training_args, overwrite=args.overwrite)
        with Pool(workers) as p:
            p.map(processing_function, dataloader)

    logging.info("DONE for %s" % args.input_folder)


if __name__ == "__main__":
    main()
