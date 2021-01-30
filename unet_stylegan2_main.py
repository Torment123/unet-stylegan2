#!/usr/bin/env python
import fire
from retry.api import retry_call
from tqdm import tqdm
from unet_stylegan2 import Trainer, NanException
from datetime import datetime
import argparse
import os
from random import random


# def main(
    # data = './data',
    # results_dir = './results',
    # models_dir = './models',
    # name = 'default',
    # new = False,
    # load_from = -1,
    # image_size = 128,
    # network_capacity = 32,
    # out_path = 'outputs/'
    # transparent = False,
    # batch_size = 12,
    # gradient_accumulate_every = 3,
    # num_train_steps = 150000,
    # learning_rate = 2e-4,
    # ttur_mult = 2,
    # num_workers =  None,
    # save_every = 1000,
    # generate = False,
    # generate_interpolation = False,
    # save_frames = False,
    # num_image_tiles = 8,
    # trunc_psi = 0.75,
    # fp16 = False,
    # no_const = False,
    # aug_prob = 0.,
    # dataset_aug_prob = 0.,
    # cr_weight = 0.5,
    # apply_pl_reg = False,
    # aug_types = ['translation', 'cutout']
# ):



def main(args):
    
    
    model = Trainer(
        name = args.name,        
        results_dir = args.results_dir,
        models_dir = args.models_dir,
        batch_size = args.batch_size,
        gradient_accumulate_every = args.gradient_accumulate_every,
        image_size = args.image_size,
        network_capacity = args.network_capacity,
        out_path = args.out_path,
        transparent = args.transparent,
        lr = args.learning_rate,
        ttur_mult = args.ttur_mult,
        num_workers = args.num_workers,
        save_every = args.save_every,
        trunc_psi = args.trunc_psi,
        fp16 = args.fp16,        
        no_const = args.no_const,
        aug_prob = args.aug_prob,
        aug_types = args.aug_types,
        dataset_aug_prob = args.dataset_aug_prob,
        cr_weight = args.cr_weight,
        apply_pl_reg = args.apply_pl_reg,
        opt = args,
        gan_type = args.gan_type,
        dataset = args.dataset,
        z_dim = args.z_dim
    )

    # if not new:
        # model.load(load_from)
    # else:
        # model.clear()

    if args.generate:
        now = datetime.now()
        timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
        samples_name = f'generated-{timestamp}'
        model.evaluate(samples_name, num_image_tiles)
        print(f'sample images generated at {results_dir}/{name}/{samples_name}')
        return

    if args.generate_interpolation:
        now = datetime.now()
        timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
        samples_name = f'generated-{timestamp}'
        model.generate_interpolation(samples_name, num_image_tiles, save_frames = save_frames)
        print(f'interpolation generated at {results_dir}/{name}/{samples_name}')
        return

    # model.set_data_src(data)

    # for _ in tqdm(range(num_train_steps - model.steps), mininterval=10., desc=f'{name}<{data}>'):
        # retry_call(model.train, tries=3, exceptions=NanException)
        # if _ % 50 == 0:
            # model.print_log()
            
    
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    model_folder_name = 'model'
    if not os.path.exists(os.path.join(args.out_path, 'model/')):
        # os.mkdir(os.path.join(args.out_path, 'model/'))
        os.mkdir(os.path.join(args.out_path, model_folder_name + '/'))
    model.opt.out_ = '%s/%s' % (args.out_path, model_folder_name)
            
    for epoch in range(args.total_epoch):
        
        model.train_epoch()
        
        # if (epoch + 1) % args.epoch_interval == 0:
            # break
        
        
     
    
    
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, default = 'coco',
                        help = 'training dataset')
    parser.add_argument('--batch_size', type = int, default = 128,
                        help = 'mini-batch size of training data. Default: 128')
    parser.add_argument('--total_epoch', type = int, default = 200,
                        help = 'number of total training epoch')
    parser.add_argument('--out_path', type = str, default = './outputs/',
                        help = 'path to output files')
    parser.add_argument('--data', type = str, default = './data')
    parser.add_argument('--results_dir', type = str, default = './results')
    parser.add_argument('--models_dir', type = str, default = './models')
    parser.add_argument('--name', type = str, default = 'default')
    parser.add_argument('--new', type = bool, default = True)
    parser.add_argument('--load_from', type = int, default = -1)
    parser.add_argument('--image_size', type = int, default = 128)
    parser.add_argument('--network_capacity', type = int, default = 32)
    parser.add_argument('--transparent', type = bool, default = False)
    parser.add_argument('--gradient_accumulate_every', type = int, default = 1)
    parser.add_argument('--num_train_steps', type = int, default = 15000)
    parser.add_argument('--learning_rate', type = float, default = 2e-4)
    parser.add_argument('--ttur_mult', type = int, default = 2)
    parser.add_argument('--num_workers', type = int, default = 0)
    parser.add_argument('--save_every', type = int, default = 1000)
    parser.add_argument('--generate', type = bool, default = False)
    parser.add_argument('--generate_interpolation', type = bool, default = False)
    parser.add_argument('--save_frames', type = bool, default = False)
    parser.add_argument('--num_image_tiles', type = int, default = 8)
    parser.add_argument('--trunc_psi', type = float, default = 0.75)
    parser.add_argument('--fp16', type = bool, default = False)
    parser.add_argument('--no_const', type = bool, default = False)
    parser.add_argument('--aug_prob', type = float, default = 0.)
    parser.add_argument('--dataset_aug_prob', type = float, default = 0.)
    parser.add_argument('--cr_weight', type = float, default = 0.5)
    parser.add_argument('--apply_pl_reg', type = bool, default = True)
    parser.add_argument('--aug_types', type = list, default = ['translation', 'cutout'])
    parser.add_argument('--gan_type', type = str, default = 'lost')
    parser.add_argument('--z_dim', type = float, default = 128, help = 'dimension of the object style latent vector')
    parser.add_argument('--idx_interval', type = int, default = 1)
    parser.add_argument('--epoch_interval', type = int, default = 2)
    parser.add_argument('--latent_dim', type = int, default = 512)
                        
    
    
    args = parser.parse_args()
    
    main(args)
    
    # fire.Fire(train_from_folder)
