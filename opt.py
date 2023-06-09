"""
This script defines the input parameters that can be customized from the command line
"""

import argparse
import datetime
import json
import os

def get_opts():
    parser=argparse.ArgumentParser()
    # input paths
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of the input dataset')
    parser.add_argument('--target_dir',type=str,required=True,
                        help="target root dir")
    parser.add_argument("--lr", type=int,default=0.0002)
    parser.add_argument("--adam_beta1",type=int,default=0.5)
    parser.add_argument("--lambda_cycle",type=int,default=10.0)
    parser.add_argument("--lambda_identity",type=int,default=0.5)
    parser.add_argument("--initialization_epochs",type=int,default=10)
    parser.add_argument("--content_loss_weight",type=int,default=5)
    parser.add_argument("--num_epochs",type=int,default=100)
    parser.add_argument("--save_path",default="./checkpoints/CycleGAN/")
    parser.add_argument("--image_test",action="store_false",
                        help="if action, generate test image per every 5 epochs")
    parser.add_argument("--generated_image_save_path",default='generated_images/CycleGAN/')
    parser.add_argument("--batch_size",type=int, default=4)
    parser.add_argument("--print_every",type=int,default=100)
    parser.add_argument("--test_image_path",default="generated_images/test")
    parser.add_argument("--model_path",type=str)
    parser.add_argument("--test",action="store_true")
    parser.add_argument("--model",default="base")
    parser.add_argument("--project_name", default="Dental_CycleGan",help="wandb Project name" )
    parser.add_argument("--run_name", help="Run_name")
    parser.add_argument("--resize",type=int, help= " If use resizing, use this. default is image input size")
    parser.add_argument("--n_res", type=int, default=6)
    parser.add_argument("--gray", action='store_true')
    
    
    args=parser.parse_args()
    
    return args