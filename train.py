# python imports
import argparse
import glob
import os
import time
import datetime
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.utils.data
# for visualization
from torch.utils.tensorboard import SummaryWriter
import wandb

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)

def get_filename(path):
    return os.path.splitext(os.path.basename(path))[0]

def init_wandb(cfg):
    
    tags = []
    dataset_tags = {'enigma_hd_hr': 'Enigma Hand Take/Release', 
                    'enigma_hc_hr': 'Enigma Hand Contact/De-Contact', 
                    'enigma_timestamp': 'Enigma only Timestamp'}
    
    dataset_filename = cfg['dataset']['json_file']
    dataset_filename = get_filename(dataset_filename)
    
    if dataset_filename in dataset_tags:
        tags.append(dataset_tags[dataset_filename])
    else:
        raise ValueError("Dataset not supported")
    
    wandb.init(
        project="actionformer-project",
        config=cfg,
        tags=tags,
    )
    
    # we consider only the last mAP value
    # wandb.define_metric("val/mAP", summary="last")
    wandb.define_metric("val/mAP", summary="best")
    
    artifact = wandb.Artifact(name='enigma_dataset', type='dataset')
    artifact.add_file(cfg['dataset']['json_file']) # Adds multiple files to artifact
    wandb.log_artifact(artifact)
    
def wandb_add_artifact_model(ckpt_filename: str):
    """ Add model checkpoint to wandb artifact

    Args:
        ckpt_filename (str): path to checkpoint file
    """    
    artifact = wandb.Artifact(name='actionformer_model', type='model')
    artifact.add_file(ckpt_filename, name='ckpt')
    wandb.log_artifact(artifact)

################################################################################
def main(args):
    """main function that handles training / inference"""

    """1. setup parameters / folders"""
    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    pprint(cfg)

    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg['output_folder']):
        os.mkdir(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output))
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    """2. create training dataset / dataloader"""
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )
    
    """2. create validation dataset / dataloader"""
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    
    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']
    
    
    # start a new wandb run to track this script
    init_wandb(cfg)

    # FOR VALIDATION
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])

    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )
    
    val_db_vars = val_dataset.get_attributes()
    det_eval = ANETdetection(
        val_dataset.json_file,
        val_dataset.split[0],
        tiou_thresholds = val_db_vars['tiou_thresholds']
    )
    
    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model)

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(args.resume,
                map_location = lambda storage, loc: storage.cuda(
                    cfg['devices'][0]))
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            # also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """4. training / validation loop"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs']
    )
    
    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema = model_ema,
            clip_grad_l2norm = cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer,
            print_freq=args.print_freq
        )

        # save ckpt once in a while (makes sense to test here on the validation set)
        if (
            ((epoch + 1) == max_epochs) or
            ((args.ckpt_freq > 0) and ((epoch + 1) % args.ckpt_freq == 0))
        ):
            
            
            # model_val = make_meta_arch(cfg['model_name'], **cfg['model'])
            # # not ideal for multi GPU training, ok for now
            # model_val = nn.DataParallel(model, device_ids=cfg['devices'])
            # model_val.load_state_dict(model_ema.module.state_dict())
            
            mAP = valid_one_epoch(
                val_loader,
                model_ema.module,
                epoch,
                evaluator=det_eval,
                output_file=None,
                ext_score_file=None,
                tb_writer=tb_writer,
                print_freq=20
            )

            save_states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            save_states['state_dict_ema'] = model_ema.module.state_dict()
            file_name_ckpt = 'epoch_{:03d}.pth.tar'.format(epoch + 1)
            save_checkpoint(
                save_states,
                False,
                file_folder=ckpt_folder,
                file_name=file_name_ckpt
            )

    print('Final average mAP: {:>4.2f} (%)'.format(mAP*100))
    
    # save only last ckpt
    ckpt_filename = os.path.join(ckpt_folder, file_name_ckpt)
    wandb_add_artifact_model(ckpt_filename)
    
    # wrap up
    tb_writer.close()
    print("All done!")
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    args = parser.parse_args()
    main(args)
