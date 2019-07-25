"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
"""
from __future__ import division
from __future__ import print_function
from builtins import range

import time
import random
import numpy as np
import json
import os
import sys
import math
import argparse
import ast
from tqdm import tqdm
import logging
from collections import defaultdict
import h5py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
import torchnet as tnt

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(DIR_PATH,'..'))

from learning import spg
from learning import graphnet
from learning import pointnet
from learning import metrics

def main():
    parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')

    # Optimization arguments
    parser.add_argument('--wd', default=0, type=float, help='Weight decay')
    parser.add_argument('--lr', default=1e-2, type=float, help='Initial learning rate')
    parser.add_argument('--lr_decay', default=0.7, type=float, help='Multiplicative factor used on learning rate at `lr_steps`')
    parser.add_argument('--lr_steps', default='[]', help='List of epochs where the learning rate is decreased by `lr_decay`')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to train. If <=0, only testing will be done.')
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size')
    parser.add_argument('--optim', default='adam', help='Optimizer: sgd|adam')
    parser.add_argument('--grad_clip', default=1, type=float, help='Element-wise clipping of gradient. If 0, does not clip')
    parser.add_argument('--loss_weights', default='none', help='[none, proportional, sqrt] how to weight the loss function')

    # Learning process arguments
    parser.add_argument('--cuda', default=1, type=int, help='Bool, use cuda')
    parser.add_argument('--nworkers', default=0, type=int, help='Num subprocesses to use for data loading. 0 means that the data will be loaded in the main process')
    parser.add_argument('--test_nth_epoch', default=1, type=int, help='Test each n-th epoch during training')
    parser.add_argument('--save_nth_epoch', default=1, type=int, help='Save model each n-th epoch during training')
    parser.add_argument('--test_multisamp_n', default=10, type=int, help='Average logits obtained over runs with different seeds')

    # Dataset
    parser.add_argument('--dataset', default='sema3d', help='Dataset name: sema3d|s3dis')
    parser.add_argument('--cvfold', default=0, type=int, help='Fold left-out for testing in leave-one-out setting (S3DIS)')
    parser.add_argument('--odir', default='results', help='Directory to store results')
    parser.add_argument('--resume', default='', help='Loads a previously saved model.')
    parser.add_argument('--db_train_name', default='train')
    parser.add_argument('--db_test_name', default='test')
    parser.add_argument('--use_val_set', type=int, default=0)
    parser.add_argument('--SEMA3D_PATH', default='datasets/semantic3d')
    parser.add_argument('--S3DIS_PATH', default='datasets/s3dis')
    parser.add_argument('--VKITTI_PATH', default='datasets/vkitti')
    parser.add_argument('--CUSTOM_SET_PATH', default='datasets/custom_set')

    # Model
    parser.add_argument('--model_config', default='gru_10,f_8', help='Defines the model as a sequence of layers, see graphnet.py for definitions of respective layers and acceptable arguments. In short: rectype_repeats_mv_layernorm_ingate_concat, with rectype the type of recurrent unit [gru/crf/lstm], repeats the number of message passing iterations, mv (default True) the use of matrix-vector (mv) instead vector-vector (vv) edge filters, layernorm (default True) the use of layernorms in the recurrent units, ingate (default True) the use of input gating, concat (default True) the use of state concatenation')
    parser.add_argument('--seed', default=1, type=int, help='Seed for random initialisation')
    parser.add_argument('--edge_attribs', default='delta_avg,delta_std,nlength/ld,surface/ld,volume/ld,size/ld,xyz/d', help='Edge attribute definition, see spg_edge_features() in spg.py for definitions.')

    # Point cloud processing
    parser.add_argument('--pc_attribs', default='xyzrgbelpsvXYZ', help='Point attributes fed to PointNets, if empty then all possible. xyz = coordinates, rgb = color, e = elevation, lpsv = geometric feature, d = distance to center')
    parser.add_argument('--pc_augm_scale', default=0, type=float, help='Training augmentation: Uniformly random scaling in [1/scale, scale]')
    parser.add_argument('--pc_augm_rot', default=1, type=int, help='Training augmentation: Bool, random rotation around z-axis')
    parser.add_argument('--pc_augm_mirror_prob', default=0, type=float, help='Training augmentation: Probability of mirroring about x or y axes')
    parser.add_argument('--pc_augm_jitter', default=1, type=int, help='Training augmentation: Bool, Gaussian jittering of all attributes')
    parser.add_argument('--pc_xyznormalize', default=1, type=int, help='Bool, normalize xyz into unit ball, i.e. in [-0.5,0.5]')

    # Filter generating network
    parser.add_argument('--fnet_widths', default='[32,128,64]', help='List of width of hidden filter gen net layers (excluding the input and output ones, they are automatic)')
    parser.add_argument('--fnet_llbias', default=0, type=int, help='Bool, use bias in the last layer in filter gen net')
    parser.add_argument('--fnet_orthoinit', default=1, type=int, help='Bool, use orthogonal weight initialization for filter gen net.')
    parser.add_argument('--fnet_bnidx', default=2, type=int, help='Layer index to insert batchnorm to. -1=do not insert.')
    parser.add_argument('--edge_mem_limit', default=30000, type=int, help='Number of edges to process in parallel during computation, a low number can reduce memory peaks.')

    # Superpoint graph
    parser.add_argument('--spg_attribs01', default=1, type=int, help='Bool, normalize edge features to 0 mean 1 deviation')
    parser.add_argument('--spg_augm_nneigh', default=100, type=int, help='Number of neighborhoods to sample in SPG')
    parser.add_argument('--spg_augm_order', default=3, type=int, help='Order of neighborhoods to sample in SPG')
    parser.add_argument('--spg_augm_hardcutoff', default=512, type=int, help='Maximum number of superpoints larger than args.ptn_minpts to sample in SPG')
    parser.add_argument('--spg_superedge_cutoff', default=-1, type=float, help='Artificially constrained maximum length of superedge, -1=do not constrain')

    # Point net
    parser.add_argument('--ptn_minpts', default=40, type=int, help='Minimum number of points in a superpoint for computing its embedding.')
    parser.add_argument('--ptn_npts', default=128, type=int, help='Number of input points for PointNet.')
    parser.add_argument('--ptn_widths', default='[[64,64,128,128,256], [256,64,32]]', help='PointNet widths')
    parser.add_argument('--ptn_widths_stn', default='[[64,64,128], [128,64]]', help='PointNet\'s Transformer widths')
    parser.add_argument('--ptn_nfeat_stn', default=11, type=int, help='PointNet\'s Transformer number of input features')
    parser.add_argument('--ptn_prelast_do', default=0, type=float)
    parser.add_argument('--ptn_mem_monger', default=1, type=int, help='Bool, save GPU memory by recomputing PointNets in back propagation.')

    args = parser.parse_args()
    args.start_epoch = 0
    args.lr_steps = ast.literal_eval(args.lr_steps)
    args.fnet_widths = ast.literal_eval(args.fnet_widths)
    args.ptn_widths = ast.literal_eval(args.ptn_widths)
    args.ptn_widths_stn = ast.literal_eval(args.ptn_widths_stn)


    print('Will save to ' + args.odir)
    if not os.path.exists(args.odir):
        os.makedirs(args.odir)
    with open(os.path.join(args.odir, 'cmdline.txt'), 'w') as f:
        f.write(" ".join(["'"+a+"'" if (len(a)==0 or a[0]!='-') else a for a in sys.argv]))

    set_seed(args.seed, args.cuda)
    logging.getLogger().setLevel(logging.INFO)  #set to logging.DEBUG to allow for more prints
    if (args.dataset=='sema3d' and args.db_test_name.startswith('test')) or (args.dataset.startswith('s3dis_02') and args.cvfold==2):
        # needed in pytorch 0.2 for super-large graphs with batchnorm in fnet  (https://github.com/pytorch/pytorch/pull/2919)
        torch.backends.cudnn.enabled = False

    # Decide on the dataset
    if args.dataset=='sema3d':
        import sema3d_dataset
        dbinfo = sema3d_dataset.get_info(args)
        create_dataset = sema3d_dataset.get_datasets
    elif args.dataset=='s3dis':
        import s3dis_dataset
        dbinfo = s3dis_dataset.get_info(args)
        create_dataset = s3dis_dataset.get_datasets
    elif args.dataset=='vkitti':
        import vkitti_dataset
        dbinfo = vkitti_dataset.get_info(args)
        create_dataset = vkitti_dataset.get_datasets
    elif args.dataset=='custom_dataset':
        import custom_dataset #<- to write!
        dbinfo = custom_dataset.get_info(args)
        create_dataset = custom_dataset.get_datasets
    else:
        raise NotImplementedError('Unknown dataset ' + args.dataset)

    # Create model and optimizer
    if args.resume != '':
        if args.resume=='RESUME': args.resume = args.odir + '/model.pth.tar'
        model, optimizer, stats = resume(args, dbinfo)
    else:
        model = create_model(args, dbinfo)
        optimizer = create_optimizer(args, model)
        stats = []

    train_dataset, test_dataset, valid_dataset = create_dataset(args)

    print('Train dataset: %i elements - Test dataset: %i elements - Validation dataset: %i elements' % (len(train_dataset),len(test_dataset),len(valid_dataset)))
    ptnCloudEmbedder = pointnet.CloudEmbedder(args)
    scheduler = MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_decay, last_epoch=args.start_epoch-1)


    ############
    def train():
        """ Trains for one epoch """
        model.train()

        loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=spg.eccpc_collate, num_workers=args.nworkers, shuffle=True, drop_last=True)
        if logging.getLogger().getEffectiveLevel() > logging.DEBUG: loader = tqdm(loader, ncols=65)

        loss_meter = tnt.meter.AverageValueMeter()
        acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
        confusion_matrix = metrics.ConfusionMatrix(dbinfo['classes'])
        t0 = time.time()

        # iterate over dataset in batches
        for bidx, (targets, GIs, clouds_data) in enumerate(loader):
            t_loader = 1000*(time.time()-t0)

            model.ecc.set_info(GIs, args.cuda)
            label_mode_cpu, label_vec_cpu, segm_size_cpu = targets[:,0], targets[:,2:], targets[:,1:].sum(1)
            if args.cuda:
                label_mode, label_vec, segm_size = label_mode_cpu.cuda(), label_vec_cpu.float().cuda(), segm_size_cpu.float().cuda()
            else:
                label_mode, label_vec, segm_size = label_mode_cpu, label_vec_cpu.float(), segm_size_cpu.float()

            optimizer.zero_grad()
            t0 = time.time()

            embeddings = ptnCloudEmbedder.run(model, *clouds_data)
            outputs = model.ecc(embeddings)

            loss = nn.functional.cross_entropy(outputs, Variable(label_mode), weight=dbinfo["class_weights"])

            loss.backward()
            ptnCloudEmbedder.bw_hook()

            if args.grad_clip>0:
                for p in model.parameters():
                    p.grad.data.clamp_(-args.grad_clip, args.grad_clip)
            optimizer.step()

            t_trainer = 1000*(time.time()-t0)
            #loss_meter.add(loss.data[0]) # pytorch 0.3
            loss_meter.add(loss.item()) # pytorch 0.4

            o_cpu, t_cpu, tvec_cpu = filter_valid(outputs.data.cpu().numpy(), label_mode_cpu.numpy(), label_vec_cpu.numpy())
            acc_meter.add(o_cpu, t_cpu)
            confusion_matrix.count_predicted_batch(tvec_cpu, np.argmax(o_cpu,1))

            logging.debug('Batch loss %f, Loader time %f ms, Trainer time %f ms.', loss.data.item(), t_loader, t_trainer)
            t0 = time.time()

        return acc_meter.value()[0], loss_meter.value()[0], confusion_matrix.get_overall_accuracy(), confusion_matrix.get_average_intersection_union()

    ############
    def eval(is_valid = False):
        """ Evaluated model on test set """
        model.eval()
        
        if is_valid: #validation
            loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, collate_fn=spg.eccpc_collate, num_workers=args.nworkers)
        else: #evaluation
            loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=spg.eccpc_collate, num_workers=args.nworkers)
            
        if logging.getLogger().getEffectiveLevel() > logging.DEBUG: loader = tqdm(loader, ncols=65)

        acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
        loss_meter = tnt.meter.AverageValueMeter()
        confusion_matrix = metrics.ConfusionMatrix(dbinfo['classes'])

        # iterate over dataset in batches
        for bidx, (targets, GIs, clouds_data) in enumerate(loader):
            model.ecc.set_info(GIs, args.cuda)
            label_mode_cpu, label_vec_cpu, segm_size_cpu = targets[:,0], targets[:,2:], targets[:,1:].sum(1).float()
            if args.cuda:
                label_mode, label_vec, segm_size = label_mode_cpu.cuda(), label_vec_cpu.float().cuda(), segm_size_cpu.float().cuda()
            else:
                label_mode, label_vec, segm_size = label_mode_cpu, label_vec_cpu.float(), segm_size_cpu.float()

            embeddings = ptnCloudEmbedder.run(model, *clouds_data)
            outputs = model.ecc(embeddings)
            
            loss = nn.functional.cross_entropy(outputs, Variable(label_mode), weight=dbinfo["class_weights"])
            loss_meter.add(loss.item()) 

            o_cpu, t_cpu, tvec_cpu = filter_valid(outputs.data.cpu().numpy(), label_mode_cpu.numpy(), label_vec_cpu.numpy())
            if t_cpu.size > 0:
                acc_meter.add(o_cpu, t_cpu)
                confusion_matrix.count_predicted_batch(tvec_cpu, np.argmax(o_cpu,1))

        return meter_value(acc_meter), loss_meter.value()[0], confusion_matrix.get_overall_accuracy(), confusion_matrix.get_average_intersection_union(), confusion_matrix.get_mean_class_accuracy()

    ############
    def eval_final():
        """ Evaluated model on test set in an extended way: computes estimates over multiple samples of point clouds and stores predictions """
        model.eval()

        acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
        confusion_matrix = metrics.ConfusionMatrix(dbinfo['classes'])
        collected, predictions = defaultdict(list), {}

        # collect predictions over multiple sampling seeds
        for ss in range(args.test_multisamp_n):
            test_dataset_ss = create_dataset(args, ss)[1]
            loader = torch.utils.data.DataLoader(test_dataset_ss, batch_size=1, collate_fn=spg.eccpc_collate, num_workers=args.nworkers)
            if logging.getLogger().getEffectiveLevel() > logging.DEBUG: loader = tqdm(loader, ncols=65)

            # iterate over dataset in batches
            for bidx, (targets, GIs, clouds_data) in enumerate(loader):
                model.ecc.set_info(GIs, args.cuda)
                label_mode_cpu, label_vec_cpu, segm_size_cpu = targets[:,0], targets[:,2:], targets[:,1:].sum(1).float()

                embeddings = ptnCloudEmbedder.run(model, *clouds_data)
                outputs = model.ecc(embeddings)

                fname = clouds_data[0][0][:clouds_data[0][0].rfind('.')]
                collected[fname].append((outputs.data.cpu().numpy(), label_mode_cpu.numpy(), label_vec_cpu.numpy()))

        # aggregate predictions (mean)
        for fname, lst in collected.items():
            o_cpu, t_cpu, tvec_cpu = list(zip(*lst))
            if args.test_multisamp_n > 1:
                o_cpu = np.mean(np.stack(o_cpu,0),0)
            else:
                o_cpu = o_cpu[0]
            t_cpu, tvec_cpu = t_cpu[0], tvec_cpu[0]
            predictions[fname] = np.argmax(o_cpu,1)
            o_cpu, t_cpu, tvec_cpu = filter_valid(o_cpu, t_cpu, tvec_cpu)
            if t_cpu.size > 0:
                acc_meter.add(o_cpu, t_cpu)
                confusion_matrix.count_predicted_batch(tvec_cpu, np.argmax(o_cpu,1))

        per_class_iou = {}
        perclsiou = confusion_matrix.get_intersection_union_per_class()
        for c, name in dbinfo['inv_class_map'].items():
            per_class_iou[name] = perclsiou[c]

        return meter_value(acc_meter), confusion_matrix.get_overall_accuracy(), confusion_matrix.get_average_intersection_union(), per_class_iou, predictions,  confusion_matrix.get_mean_class_accuracy(), confusion_matrix.confusion_matrix

    ############
    # Training loop
    try:
        best_iou = stats[-1]['best_iou']
    except:
        best_iou = 0
    TRAIN_COLOR = '\033[0m'
    VAL_COLOR = '\033[0;94m' 
    TEST_COLOR = '\033[0;93m'
    BEST_COLOR = '\033[0;92m'
    epoch = args.start_epoch
    
    for epoch in range(args.start_epoch, args.epochs):
        print('Epoch {}/{} ({}):'.format(epoch, args.epochs, args.odir))
        scheduler.step()

        acc, loss, oacc, avg_iou = train()

        print(TRAIN_COLOR + '-> Train Loss: %1.4f   Train accuracy: %3.2f%%' % (loss, acc))

        new_best_model = False
        if args.use_val_set:
            acc_val, loss_val, oacc_val, avg_iou_val, avg_acc_val = eval(True)
            print(VAL_COLOR + '-> Val Loss: %1.4f  Val accuracy: %3.2f%%  Val oAcc: %3.2f%%  Val IoU: %3.2f%%  best ioU: %3.2f%%' % \
                 (loss_val, acc_val, 100*oacc_val, 100*avg_iou_val,100*max(best_iou,avg_iou_val)) + TRAIN_COLOR)
            if avg_iou_val>best_iou: #best score yet on the validation set
                print(BEST_COLOR + '-> New best model achieved!' + TRAIN_COLOR)
                best_iou = avg_iou_val
                new_best_model = True
                torch.save({'epoch': epoch + 1, 'args': args, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()},
                       os.path.join(args.odir, 'model.pth.tar'))
        elif epoch % args.save_nth_epoch == 0 or epoch==args.epochs-1:
                torch.save({'epoch': epoch + 1, 'args': args, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()},
                       os.path.join(args.odir, 'model.pth.tar'))
        #test every test_nth_epochs
        #or test after each enw model (but skip the first 5 for efficiency)
        if (not(args.use_val_set) and (epoch+1) % args.test_nth_epoch == 0)  \
           or (args.use_val_set and new_best_model and epoch > 5): 
            acc_test, loss_test, oacc_test, avg_iou_test, avg_acc_test = eval(False)
            print(TEST_COLOR + '-> Test Loss: %1.4f  Test accuracy: %3.2f%%  Test oAcc: %3.2f%%  Test avgIoU: %3.2f%%' % \
                 (loss_test, acc_test, 100*oacc_test, 100*avg_iou_test) + TRAIN_COLOR)
        else:
            acc_test, loss_test, oacc_test, avg_iou_test, avg_acc_test = 0, 0, 0, 0, 0

        stats.append({'epoch': epoch, 'acc': acc, 'loss': loss, 'oacc': oacc, 'avg_iou': avg_iou, 'acc_test': acc_test, 'oacc_test': oacc_test, 'avg_iou_test': avg_iou_test, 'avg_acc_test': avg_acc_test, 'best_iou' : best_iou})

        if epoch % args.save_nth_epoch == 0 or epoch==args.epochs-1:
            with open(os.path.join(args.odir, 'trainlog.json'), 'w') as outfile:
                json.dump(stats, outfile,indent=4)
            torch.save({'epoch': epoch + 1, 'args': args, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(), 'scaler': scaler},
                       os.path.join(args.odir, 'model.pth.tar'))
        
        if math.isnan(loss): break
    
    if len(stats)>0:
        with open(os.path.join(args.odir, 'trainlog.json'), 'w') as outfile:
            json.dump(stats, outfile, indent=4)

    if args.use_val_set :
        args.resume = args.odir + '/model.pth.tar'
        model, optimizer, stats = resume(args, dbinfo)
        torch.save({'epoch': epoch + 1, 'args': args, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()},
                       os.path.join(args.odir, 'model.pth.tar'))
    
    # Final evaluation
    if args.test_multisamp_n>0 and args.db_test_name == 'test':
        acc_test, oacc_test, avg_iou_test, per_class_iou_test, predictions_test, avg_acc_test, confusion_matrix = eval_final()
        print('-> Multisample {}: Test accuracy: {}, \tTest oAcc: {}, \tTest avgIoU: {}, \tTest mAcc: {}'.format(args.test_multisamp_n, acc_test, oacc_test, avg_iou_test, avg_acc_test))
        with h5py.File(os.path.join(args.odir, 'predictions_'+args.db_test_name+'.h5'), 'w') as hf:
            for fname, o_cpu in predictions_test.items():
                hf.create_dataset(name=fname, data=o_cpu) #(0-based classes)
        with open(os.path.join(args.odir, 'scores_'+args.db_test_name+'.json'), 'w') as outfile:
            json.dump([{'epoch': args.start_epoch, 'acc_test': acc_test, 'oacc_test': oacc_test, 'avg_iou_test': avg_iou_test, 'per_class_iou_test': per_class_iou_test, 'avg_acc_test': avg_acc_test}], outfile)
        np.save(os.path.join(args.odir, 'pointwise_cm.npy'), confusion_matrix)

def resume(args, dbinfo):
    """ Loads model and optimizer state from a previous checkpoint. """
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    
    checkpoint['args'].model_config = args.model_config #to ensure compatibility with previous arguments convention
    #this should be removed once new models are uploaded
    
    model = create_model(checkpoint['args'], dbinfo) #use original arguments, architecture can't change
    optimizer = create_optimizer(args, model)
    
    #model.load_state_dict(checkpoint['state_dict'])
    #to ensure compatbility of previous trained models with new InstanceNormD behavior comment line below and uncomment line above if not using our trained  models
    model.load_state_dict({k:checkpoint['state_dict'][k] for k in checkpoint['state_dict'] if k not in ['ecc.0._cell.inh.running_mean','ecc.0._cell.inh.running_var','ecc.0._cell.ini.running_mean','ecc.0._cell.ini.running_var']})

    if 'optimizer' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer'])
    for group in optimizer.param_groups: group['initial_lr'] = args.lr
    args.start_epoch = checkpoint['epoch']
    try:
        stats = json.loads(open(os.path.join(os.path.dirname(args.resume), 'trainlog.json')).read())
    except:
        stats = []
    return model, optimizer, stats
    
def create_model(args, dbinfo):
    """ Creates model """
    model = nn.Module()

    nfeat = args.ptn_widths[1][-1]
    model.ecc = graphnet.GraphNetwork(args.model_config, nfeat, [dbinfo['edge_feats']] + args.fnet_widths, args.fnet_orthoinit, args.fnet_llbias,args.fnet_bnidx, args.edge_mem_limit)

    model.ptn = pointnet.PointNet(args.ptn_widths[0], args.ptn_widths[1], args.ptn_widths_stn[0], args.ptn_widths_stn[1], dbinfo['node_feats'], args.ptn_nfeat_stn, prelast_do=args.ptn_prelast_do)

    print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
    print(model)    
    if args.cuda: 
        model.cuda()
    return model 

def create_optimizer(args, model):
    if args.optim=='sgd':
        return optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    elif args.optim=='adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

def set_seed(seed, cuda=True):
    """ Sets seeds in all frameworks"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: 
        torch.cuda.manual_seed(seed)    

def filter_valid(output, target, other=None):
    """ Removes predictions for nodes without ground truth """
    idx = target!=-100
    if other is not None:
        return output[idx,:], target[idx], other[idx,...]
    return output[idx,:], target[idx]
    
def meter_value(meter):   
    return meter.value()[0] if meter.n>0 else 0


if __name__ == "__main__": 
    main()
