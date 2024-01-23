import argparse
import math
import copy
import json
import os
import os.path as osp

import cv2
import torch
import numpy as np

from hawp.base import (
    WireframeGraph,
    show,
)
from hawp.fsl.config import cfg as model_config
from hawp.ssl.models import MODELS
import flask

from flask import Flask,request,jsonify

def dot_product(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

def magnitude(v):
    return math.sqrt(sum(x**2 for x in v))

def angle_between_vectors(v1, v2):
    dot_prod = dot_product(v1, v2)
    mag_v1 = magnitude(v1)
    mag_v2 = magnitude(v2)

    # 避免浮点数精度问题，确保余弦值在[-1, 1]范围内
    cos_theta = min(1.0, max(-1.0, dot_prod / (mag_v1 * mag_v2)))

    # 计算夹角（弧度）
    angle_rad = math.acos(cos_theta)

    # 转换为角度
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg

def trirm( wireframe,indices,angle,threshold):

        nodes = wireframe['juncs_pred']
        line_score = wireframe['lines_score'] > threshold
        lines = indices[line_score]
        # lines = indices

        if isinstance(nodes, torch.Tensor):
            nodes = nodes.cpu().numpy()
        if isinstance(lines, torch.Tensor):
            lines = lines.cpu().numpy()
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()

        delines = []
        for i in range(0, len(nodes)):
            buf = []
            ls = []
            for line in lines:
                if line[0] == i:
                    buf.append(line)

            for l in buf:
                n1 = nodes[l[0]]
                n2 = nodes[l[1]]
                lenth = (n2[1] - n1[1])**2 + (n2[0] - n1[0])**2
                lenth = math.sqrt(lenth)
                ls.append((l[0],l[1],lenth))

            idx = 0
            for (n1,n2,lenth) in ls:
                for (nn1,nn2,ll) in ls:
                    (x1,y1) = nodes[n1]
                    (x2,y2) = nodes[n2]
                    (xx1,yy1) = nodes[nn1]
                    (xx2,yy2) = nodes[nn2]
                    cta = angle_between_vectors([x2-x1,y2-y1],[xx2-xx1,yy2-yy1])
                    # print(cta)
                    if cta < angle and lenth > ll:
                        delines.append([n1,n2])
                idx += 1

        for i in range(0, len(nodes)):
            buf = []
            ls = []
            for line in lines:
                if line[1] == i:
                    buf.append(line)

            for l in buf:
                n1 = nodes[l[0]]
                n2 = nodes[l[1]]
                lenth = (n2[1] - n1[1])**2 + (n2[0] - n1[0])**2
                lenth = math.sqrt(lenth)
                ls.append((l[0],l[1],lenth))

            idx = 0
            for (n1,n2,lenth) in ls:
                for (nn1,nn2,ll) in ls:
                    (x1,y1) = nodes[n1]
                    (x2,y2) = nodes[n2]
                    (xx1,yy1) = nodes[nn1]
                    (xx2,yy2) = nodes[nn2]
                    cta = angle_between_vectors([x2-x1,y2-y1],[xx2-xx1,yy2-yy1])
                    # print(cta)
                    if cta < angle and lenth > ll:
                        delines.append([n1,n2])
                idx += 1

        lines = lines.tolist()

        for bl in delines:
            if bl in lines:
                lines.remove(bl)

        segs = []
        for (n1,n2) in lines:
            seg = [nodes[n1][0],nodes[n1][1],nodes[n2][0],nodes[n2][1]]
            segs.append(seg)
        #[x1,y1,x2,y2]
        segs = np.array(segs)
        indices = torch.tensor(np.array(lines))
        # print(segs)
        return (segs,indices)


def model_load():
    model_config.merge_from_file(os.path.join(
        'hawp','ssl','config','hawpv3.yaml'))
    model = MODELS['HAWP'](model_config, gray_scale=True)
    model = model.eval().to('cpu')
    weight_path = 'checkpoints/hawpv3-imagenet-03a84.pth'
    state_dict = torch.load(weight_path,map_location='cpu')
    model.load_state_dict(state_dict)
    print('model loaded')
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
            # print(user_input)
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

            # mirror for ansa process
            outputs['juncs_pred'][:, 1] = outputs['height'] - outputs['juncs_pred'][:, 1]
            outputs['lines_pred'][:, [1, 3]] = outputs['height'] - outputs['lines_pred'][:, [1, 3]]

            wireframe = WireframeGraph(outputs['juncs_pred'], outputs['juncs_score'], new_idcs, outputs['lines_score'], outputs['width'], outputs['height'])
            with open(json_file,'w') as f:
                json.dump(wireframe.jsonize(),f)
            if os.path.exists(order_path):
                os.remove(order_path)

app = Flask(__name__)
model = model_load()

@app.route('/', methods=['POST'])
def process_json():
    data = request.get_json()
    print('got json')
    fname = data['file']
    angle = data['cta']
    threshou = data['bar']
    image,meta = image_a(fname,512,512)
    with torch.no_grad():
        outputs,_ = model(image,[meta])
        print('got output')
        indices = WireframeGraph.xyxy2indices(outputs['juncs_pred'],outputs['lines_pred'])
        (segs,new_idcs) = trirm(outputs,indices,float(angle),float(threshou))

    # mirror for ansa process
    outputs['juncs_pred'][:, 1] = outputs['height'] - outputs['juncs_pred'][:, 1]
    outputs['lines_pred'][:, [1, 3]] = outputs['height'] - outputs['lines_pred'][:, [1, 3]]

    wireframe = WireframeGraph(outputs['juncs_pred'], outputs['juncs_score'],new_idcs, outputs['lines_score'], outputs['width'], outputs['height'])
    return wireframe.jsonize()


if __name__ == '__main__':
    app.run(debug=True)
