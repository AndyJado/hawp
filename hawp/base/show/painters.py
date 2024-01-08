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


class HAWPainter:
    line_width = None
    marker_size = None
    confidence_threshold = 0.05

    def __init__(self):

        if self.line_width is None:
            self.line_width = 2
        
        if self.marker_size is None:
            self.marker_size = max(1, int(self.line_width * 0.5))

    



    def draw_wireframe(self, ax, line_segments, *,
            edge_color = None, vertex_color = None):
        if line_segments is None:
            return
        
        if edge_color is None:
            edge_color = 'b'
        if vertex_color is None:
            vertex_color = 'c'
        
        # [x1,y1,x2,y2]
        ax.plot([line_segments[:,0],line_segments[:,2]],[line_segments[:,1],line_segments[:,3]],'-',color=edge_color)
        ax.plot(line_segments[:,0],line_segments[:,1],'.',color=vertex_color)
        ax.plot(line_segments[:,2],line_segments[:,3],'.',
        color=vertex_color)
        idx = 1
        for line in line_segments:
            idx += 1
            x = (line[0] + line[2]) / 2
            y = (line[1] + line[3]) / 2
            ax.text(x,y,str(idx),color='red')
            # print(idx)

    def trianglerm(self, wireframe):

        nodes = wireframe['juncs_pred']
        lines = wireframe['juncs_score'][wireframe['lines_score']>self.confidence_threshold]

        llll = wireframe['lines_pred'][wireframe['lines_score']>self.confidence_threshold]

        print(lines[0],llll[0])

        print(len(lines),len(llll))

        if isinstance(nodes, torch.Tensor):
            nodes = nodes.cpu().numpy()
        if isinstance(lines, torch.Tensor):
            lines = lines.cpu().numpy()

        # print(len(nodes),nodes)
        # print(len(lines),lines)

        # y/x
        new_lines = []
        for i in range(0, len(nodes)):
            buf = []
            ls = []
            for line in lines:
                if line[0] == i:
                    buf.append(line)

            for l in buf:
                n1 = nodes[l[0]]
                n2 = nodes[l[1]]
                k = (n2[1] - n1[1]) / (n2[0] - n1[0])
                lenth = (n2[1] - n1[1])**2 + (n2[0] - n1[0])**2
                lenth = math.sqrt(lenth)
                ls.append((l[0],l[1],k,lenth))

            idx = 0
            for (n1,n2,k,lenth) in ls:
                for (_,_,kk,ll) in ls:
                    diffk = abs(k - kk)
                    if diffk < 0.01 and ll > lenth:
                        new_lines.append([n1,n2])
                idx += 1

        lines = lines.tolist();

        for bl in new_lines:
            if bl in lines:
                lines.remove(bl)

        segs = []
        for (n1,n2) in lines:
            seg = [nodes[n1][0],nodes[n1][1],nodes[n2][0],nodes[n2][1]]
            segs.append(seg)
        #[x1,y1,x2,y2]
        segs = np.array(segs)
        print(segs)
        return segs
