import logging

import numpy as np
import torch 
import math


try:
    import matplotlib
    import matplotlib.animation
    import matplotlib.collections
    import matplotlib.patches
except ImportError:
    matplotlib = None


LOG = logging.getLogger(__name__)

import math

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



class HAWPainter:
    line_width = None
    marker_size = None
    confidence_threshold = 0.05

    def __init__(self):

        if self.line_width is None:
            self.line_width = 2
        
        if self.marker_size is None:
            self.marker_size = max(1, int(self.line_width * 0.5))
    

    def draw_segs(self, ax, line_segments, *,
            edge_color = None, vertex_color = None):
        if line_segments is None:
            return
        
        if edge_color is None:
            edge_color = 'b'
        if vertex_color is None:
            vertex_color = 'c'
        
        # [x1,y1,x2,y2]
        for (x1,y1,x2,y2) in line_segments:
            ax.plot([x1,x2],[y1,y2],'-',color=edge_color)
        ax.plot(line_segments[:,0],line_segments[:,1],'.',color=vertex_color)
        ax.plot(line_segments[:,2],line_segments[:,3],'.',color=vertex_color)
        idx = 0
        for line in line_segments:
            idx += 1
            x = (line[0] + line[2]) / 2
            y = (line[1] + line[3]) / 2
            ax.text(x,y,str(idx),color='brown',horizontalalignment='center',verticalalignment='center',fontsize=18)
            # print(idx)

    def trianglerm(self, wireframe,indices,kbar):

        nodes = wireframe['juncs_pred']
        lines = indices[wireframe['lines_score']>self.confidence_threshold]

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
                    if cta < kbar and lenth > ll:
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
                    if cta < kbar and lenth > ll:
                        delines.append([n1,n2])
                idx += 1

        lines = lines.tolist();

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
