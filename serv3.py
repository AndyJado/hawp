import torch
import random 
import numpy as np
import os
import os.path as osp
from tqdm import tqdm

from hawp.base import to_device, setup_logger, MetricLogger, save_config, show, WireframeGraph


from hawp.fsl.solver import make_lr_scheduler, make_optimizer
from hawp.fsl.config import cfg as model_config

from hawp.ssl.config import Config, load_config
from hawp.ssl.datasets import dataset_util
from hawp.ssl.models import MODELS

from torch.utils.data import DataLoader
import torch.utils.data.dataloader as torch_loader

from pathlib import Path
import argparse, yaml, logging, time, datetime, cv2, copy, sys, json

def model_load():
    model_config.merge_from_file(os.path.join(
        'hawp','ssl','config','hawpv3.yaml'))
    model = MODELS['HAWP'](model_config, gray_scale=True)
    model = model.eval().to('cpu')
    weight_path = 'checkpoints/hawpv3-imagenet-03a84.pth'
    state_dict = torch.load(weight_path,map_location='cpu')
    model.load_state_dict(state_dict)
    return model

def image_a(fname,width,height):
    image = cv2.imread(fname,0)
    
    ori_shape = image.shape[:2]
    image_cp = copy.deepcopy(image)
    # fix sized input!
    image_ = cv2.resize(image_cp,(width,height))
    image_ = torch.from_numpy(image_).float()/255.0
    image_ = image_[None,None].to('cpu')
    meta = {
        'width': ori_shape[1],
        'height':ori_shape[0],
        'filename': ''
    }
    return image_,meta

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--np', default=1,type=int)
    args = parser.parse_args()
    return args

def main():
    script_path = os.path.abspath(__file__)
    # Change the current working directory to the directory containing the script
    script_directory = os.path.dirname(script_path)
    os.chdir(script_directory)
    args = parse_args()
    model = model_load()
    order_path = 'orders/' + str(args.np) + '.txt'

    with torch.no_grad():
        while True:
            if not os.path.exists(order_path):
                continue
            with open(order_path,'r') as order:
                user_input = order.readline()
            if not user_input:
                continue
            print(user_input)
            data = json.loads(user_input.strip())
            fname = data['file']
            outy = data['outy']
            cta = data['cta']
            bar = data['bar']
            show.painters.HAWPainter.confidence_threshold = bar
            painter = show.painters.HAWPainter()
            image,meta = image_a(fname,512,512)
            outputs,_ = model(image,[meta])
            out_fig = outy + '.png'
            out_json = outy + '.json'
            fig_file = osp.join(outy)
            json_file = osp.join(out_json)
            indices = WireframeGraph.xyxy2indices(outputs['juncs_pred'],outputs['lines_pred'])
            with show.image_canvas(fname, fig_file=fig_file) as ax:
                (segs,new_idcs) = painter.trianglerm(outputs,indices,cta)
                # print('new',new_idcs,len(new_idcs),'\n')
                painter.draw_segs(ax,segs,False)
            wireframe = WireframeGraph(outputs['juncs_pred'], outputs['juncs_score'], new_idcs, outputs['lines_score'], outputs['width'], outputs['height'])
            with open(json_file,'w') as f:
                json.dump(wireframe.jsonize(),f)
            if os.path.exists(order_path):
                os.remove(order_path)


if __name__ == '__main__':
    main()
