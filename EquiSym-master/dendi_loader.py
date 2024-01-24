import os
import torch
import random
import numpy as np
import cv2
import json
from torch.utils.data import Dataset
import PIL
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms

# from matplotlib.collections import PatchCollection
# from matplotlib.patches import Polygon

from utils import *
from shapely.geometry import Polygon
from shapely import affinity
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DrawEllipse(object):
    def __init__(self, cuts=120):
        self.cuts = cuts

    def make_ellipse(self, w, h, cuts):
        points = []
        
        for i in range(cuts):
            deg = i*(360/cuts)*3.14/180
            points.append((w*np.sin(deg), h*np.cos(deg)))
        return Polygon(points)

    def draw_ellipse(self, points, size, fill=1):
        axis = Image.new('L', size)
        # w, h = img.size
        draw = ImageDraw.Draw(axis)
            
        draw.polygon(points, fill=fill, outline=None)
        axis = np.asarray(axis).astype(np.float32)
        return axis#, theta_degree

    def dist(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        dist = (x1 - x2) ** 2 + (y1 - y2) **2
        return dist ** 0.5

    def get_theta(self, point1, point2):
        b, a = (point2[1] - point1[1]), (point2[0] - point1[0])
        if a == 0:
            # print('line |')
            return 90
        else:
            tan = - b / a # i changed this too
        return np.arctan(tan) * 180 / np.pi
        
    def __call__(self, points, size, cuts=120):
        if len(points) == 4:
            ctr_x = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4
            ctr_y = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4
            _points = [(ctr_x, ctr_y)]
            _points += points
            points = _points
        # elif len(points) != 5:
        #     print('sth wrong ,', len(points))
            
        up, down, left, right = 1, 2, 3, 4 #point[0]为圆的中心
        left, right, up, down = points[left], points[right], points[up], points[down]
        w = (self.dist(points[0], left) + self.dist(points[0], right)) / 2
        h = (self.dist(points[0], up) + self.dist(points[0], down)) / 2
        theta = (self.get_theta(up, down) + 180) % 180 - 90
        
        ellipse = self.make_ellipse(w, h, cuts)
        ellipse = affinity.translate(ellipse, xoff=points[0][0], yoff=points[0][1])
        ellipse = affinity.rotate(ellipse, -theta, origin='centroid')
        
        axis = self.draw_ellipse(ellipse.exterior.coords, size)
        return axis, (points[0][0], points[0][1])
        #(points[0][0], points[0][1])为圆的中心x和y

class NewSymmetryDatasetsBase(Dataset):
    def __init__(self, sym_type, get_polygon=2, split='train', root='./sym_datasets/DENDI', with_ref_circle=1, n_theta=8):
        super(NewSymmetryDatasetsBase, self).__init__()
        self.root = root
        self.split = split
        self.sym_type = sym_type
        self.get_polygon = get_polygon
        self.with_ref_circle = with_ref_circle
        self.order_list = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17, 20, 22, 26, 28, 29] # 20+1
        self.img_list, self.gt_list = self.get_data_list()
        self.ellipse = DrawEllipse()
        self.n_theta = n_theta
        self.ellipse_theta_filter = self.construct_theta_filter()

    def construct_theta_filter(self):
        # angle_interval, n_theta = 45, 8
        angle_interval = int(360 / self.n_theta) #角度间隔：360/8=45°
        n_theta = self.n_theta #8
        if self.split == 'test':
            c = int(417*5)
            # c = int(417 * 1.6)
        else: #训练集train为这里
            c = int(417 * 1.6)
        self.filter_ks = c
        base = torch.ones((c*2+1, c*2+1))
        indices_all = torch.nonzero(base)
        center = torch.tensor([c, c])
        dh_dw = indices_all - center
        tangents = - (dh_dw[:, 0]) / (dh_dw[:, 1] + 1e-2)
        theta = np.arctan(tangents)
        theta = (theta * 180 / np.pi) % 360
        t_lbl = torch.zeros(c*2+1, c*2+1, n_theta)

        d = angle_interval / 2
        k = theta // d
        a = k + 1 - theta / d
        indices1, indices2 = (k + 3) % n_theta, (k + 4) % n_theta

        t_lbl[indices_all[:, 0], indices_all[:, 1], indices1.long()] = a
        t_lbl[indices_all[:, 0], indices_all[:, 1], indices2.long()] = 1 - a

        return t_lbl

    def get_data_list(self):#获取图片，pt里面存的是图片的路径名字
        split_dict = torch.load(os.path.join(self.root, '%s_split.pt') % self.sym_type)
        if self.split == 'all':
            split_dict = split_dict['all']
            img_key, ann_key = 'img', 'ann'
        else:
            img_key, ann_key  = '%s_img' % self.split, '%s_ann' % self.split
        img_list = [os.path.join(self.root, name) for name in split_dict[img_key]]
        gt_list = []

        for path in split_dict[ann_key]:
            json_path = os.path.join(self.root, path)
            with open(json_path) as json_file:
                json_data = json.load(json_file)
                gt = json_data['figures']#figures里面存了对称轴坐标
                gt_list.append(gt)        

        return img_list, gt_list
        #img_list存储图像路径，gt_list存储图像标注json文件路径
    
    def process_data_ref(self, gt, size):
        gt_dict = {}
        lines = []
        ellipse_pts = []
        
        for f in gt:
            if f['label'] in ['reflection']: # polyline, non circle
                for i in range(len(f['shape']['coordinates']) - 1): #coordinates里面存的就是对称轴坐标
                    x1, y1 = f['shape']['coordinates'][i]
                    x2, y2 = f['shape']['coordinates'][i+1]
                    lines.append([x1, y1, x2, y2])
            elif f['label'] in ['reflection-circle']:
                if len(f['shape']['coordinates']) == 2:
                    x1, y1 = f['shape']['coordinates'][0]
                    x2, y2 = f['shape']['coordinates'][1]
                    lines.append([x1, y1, x2, y2])
                else:
                    ellipse_pts.append(f['shape']['coordinates'])#反射对称图像为圆形

        ellipse_axis_lbl = []
        ellipse_coords = []

        if self.with_ref_circle in [1]: #这里为ture
            for i, pts in enumerate(ellipse_pts):
                ellipse_axis, center_coords = self.ellipse(pts, size)
                #ellipse_axis是画出对称的整个圆的得分图，center_coords是圆的中心
                _ellipse_axis_lbl = ellipse_axis * (i + 1000 + 1)
                # _ellipse_axis_lbl = - ellipse_axis * (i+1+axis_lbl.max())
                ellipse_axis_lbl.append(_ellipse_axis_lbl)
                # (cx, cy, cs, cy)
                _coords = [center_coords[0] / size[0], center_coords[1] / size[1], \
                    center_coords[0] / size[0], center_coords[1] / size[1] ]
                
                ellipse_coords.append(_coords)
        elif self.with_ref_circle in [2]:
            for i, pts in enumerate(ellipse_pts):
                if len(pts) == 4:
                    lines.append([pts[0][0], pts[0][1], pts[1][0], pts[1][1]])
                    lines.append([pts[2][0], pts[2][1], pts[3][0], pts[3][1]])
                else:
                    lines.append([pts[1][0], pts[1][1], pts[2][0], pts[2][1]])
                    lines.append([pts[3][0], pts[3][1], pts[4][0], pts[4][1]])

        axis_lbl, line_coords = draw_axis(lines, size) #lines为对称轴，size为图片大小
        # 返回的axis_lbl是一张对称轴图，背景黑色，白色为对称轴
        # 返回的line_coords是每条对称轴的x/宽和y/高（x,y）所占比例

        if len(ellipse_axis_lbl):
            gt_dict['ellipse_axis_lbl'] = ellipse_axis_lbl
            ellipse_axis_lbl.append(axis_lbl) #在ellipse_axis_lbl末尾加axis_lbl
            maps = np.stack(ellipse_axis_lbl, axis=0) #将线形和圆形对称图拼接
            axis_lbl = np.max(maps, axis=0) # ellipse -1 #找每列最大的

        if len(line_coords) and len(ellipse_coords):
            line_coords = np.concatenate((np.array(line_coords), np.array(ellipse_coords)), axis=0)
            #将两者的对称轴x/宽和y/高拼接
        elif len(ellipse_coords):
            line_coords = np.array(ellipse_coords)

        gt_dict['axis_lbl'] = axis_lbl #对称轴得分图
        gt_dict['axis'] = (axis_lbl > 0).astype(axis_lbl.dtype)
        #dtype返回数组中元素的数据类型，astype() 对数据类型进行转换
        #axis_lbl中大于0（对称轴）的数为1，小于0或等于的数为0 ---->则对称轴为前景类1，背景类为0
        gt_dict['line_coords'] = line_coords #对称轴的坐标x/宽和y/高
        
        return gt_dict

    def process_order(self, N):
        if N in self.order_list:
            return self.order_list.index(N)
        else:
            return 1

    def process_data_rot(self, gt, size):
        w, h = size
        gt_dict = {}
        centers = []
        orders = []

        for f in gt:
            if f['label'] in ['rotation-polygon']:
                x1, y1 = f['shape']['coordinates'][0]
                centers.append((int(x1), int(y1)))
                N = abs(int(f['attributes'][0]['value']))
                N = self.process_order(N)
                orders.append(N)
            elif f['label'] in ['rotation-circle']:
                x1, y1 = f['shape']['coordinates'][0]
                centers.append((int(x1), int(y1)))
                N = abs(int(f['attributes'][0]['value']))
                N = self.process_order(N) 
                orders.append(N)

        if len(centers):
            maps = draw_points(centers, orders, size)
            maps = np.stack(maps, axis=0)
            center_map = (np.sum(maps, axis=0) > 0).astype(np.float32)
            order_map = np.max(maps, axis=0)
        else:
            center_map = np.zeros((h, w)).astype(np.float32)
            order_map = np.zeros((h, w)).astype(np.float32)
        
        gt_dict['order_map'] = order_map
        gt_dict['axis_map'] = center_map
        return gt_dict

    def process_theta_ref(self, axis_lbl, axis_coords):#axis_lbl是标注对称轴坐标的数据，不同对称轴填充数字不一样
        im_h, im_w = axis_lbl.shape[-2], axis_lbl.shape[-1] #图像的高和宽
        a_lbl = torch.zeros_like(axis_lbl).unsqueeze(0).unsqueeze(1).expand(-1, self.n_theta, -1, -1)
        #变成8维度1*1*8*h*w
        ellipse_mask = (axis_lbl > 1000).float() #圆形对称轴的掩码 1为对称轴前景类，0为背景类
        ellipse_a_lbl = torch.zeros_like(axis_lbl).unsqueeze(-1).expand(-1, -1, self.n_theta)

        if axis_lbl.max() > 1000: #圆形的对称轴
            # ellipse_lbl = axis_lbl * ellipse_mask
            num_ellipse = axis_lbl.max() - 1000 #圆形对称轴的个数
            ellipse_masks = [] #圆形对称轴的掩码
            ellipse_b_lbl = [] #圆形对称轴的
            ellipse_n_pix = [] #圆形对称轴的对称轴位置为1的个数求和
            
            n = num_ellipse.long()#torch.long():向下取整
            for i in range(n):#圆形存储在后面，所以从后面开始找
                cx, cy = axis_coords[-n+i][0], axis_coords[-n+i][1] #x/宽，y/高
                b_lbl = (axis_lbl == (i + 1000 + 1)).float() #转化为浮点数，若为对称轴前景类则为1，不为对称轴背景类为0
                n_pix = b_lbl.sum() #将所有对称轴所在位置（为1）求和
                ellipse_masks.append(b_lbl) #圆形对称轴的掩码
                ellipse_n_pix.append(n_pix) #求和
                cx, cy, ks = int(cx * im_w), int(cy * im_h), int(self.filter_ks) #还原x和y的坐标，self.filter_ks=int(417 * 1.6)
                _filter = self.ellipse_theta_filter[ks - cy:ks - cy + im_h, \
                                                    ks - cx:ks - cx + im_w, :]
                b_lbl = _filter * b_lbl.unsqueeze(-1)
                ellipse_b_lbl.append(b_lbl.float())
            
            stack_order = torch.argsort(torch.tensor(ellipse_n_pix), descending=True)
            for idx in stack_order:
                ellipse_a_lbl = ellipse_a_lbl * (1 - ellipse_masks[idx].unsqueeze(-1)) \
                    + ellipse_b_lbl[idx] * ellipse_masks[idx].unsqueeze(-1)
                
            axis_lbl = axis_lbl * (1 - ellipse_mask)
        num_lines = axis_lbl.max().long() #对称轴可能的个数

        if num_lines > 0:
            ### input axis_lbl (0 ~ N), # axis_coords (x1, y1, x2, y2) normalized
            ### output axis (0 or 1) # angle label (H, W, nangle) sum 1
            axis_coords = np.array(axis_coords[:num_lines])#获取所有的对称轴
            ### y in image coordinate is different from the world coord
            theta = np.where(axis_coords[:, 2] == axis_coords[:, 0], 90, \
                            np.arctan(-(axis_coords[:, 3] - axis_coords[:, 1]) / (axis_coords[:, 2] - axis_coords[:, 0])) \
                                * 180 / np.pi )
            #np.where为三元表达式，若两个坐标的x相同则为90°，否则通过arctan(y2-y1/x2-x1)来计算对称轴的角度
            # kernel theta interval
            d = self.angle_interval / 2 #d=45/2
            k = theta // d
            a = k + 1 - theta / d
            indices1, indices2 = (k + 3) % self.n_theta, (k + 4) % self.n_theta
            # indices1, indices2 = (k) % self.n_theta, (k + 1) % self.n_theta
            # a_lbl [dummy 0, kernel1, kernel2, ...]
            a_lbl = np.zeros((theta.shape[0] + 1, self.n_theta), dtype=np.float32)
            a_lbl[np.arange(theta.shape[0]) + 1, indices1.astype(np.uint8)] = a
            a_lbl[np.arange(theta.shape[0]) + 1, indices2.astype(np.uint8)] = 1 - a
            # a_lbl (H, W, nangle)

            a_lbl = a_lbl[axis_lbl.int(), :]#将真实对称轴的方向量化为0-1编码
            a_lbl = torch.from_numpy(a_lbl).permute(2, 0, 1).unsqueeze(0)
            #这个量化结果就是有8个方向的数组，如果该对称轴在该方向，则该方向对应位置为1
        
        ellipse_a_lbl = ellipse_a_lbl.permute(2, 0, 1).unsqueeze(0)
        a_lbl = torch.where(axis_lbl > 0, \
                            a_lbl, ellipse_a_lbl)
        ### move max_pool to model (GPU, ver.) for training
        # a_lbl = F.max_pool2d(a_lbl, kernel_size=5, stride=1, padding=2)
        a_lbl = F.interpolate(a_lbl, (im_h, im_w), mode='nearest') #数组采样，把图片变为图像大小
        return a_lbl

    def process_theta_rot(self, a_lbl):
        # a_lbl (order list index 0~N), out of bound -> 1 
        # draw_points conver 0 -> 255
        a_lbl = a_lbl.unsqueeze(0)
        ### move max_pool to model (GPU, ver.) for training
        # a_lbl = F.max_pool2d(a_lbl, kernel_size=5, stride=1, padding=2).squeeze(1).squeeze(0)
        fg_mask = (a_lbl > 0).float()
        # a_lbl (255->0, 1, 2, ..., N-1)
        a_lbl = (a_lbl != 255).float() * a_lbl
        # a_lbl ((0, 255)->1, 1->2, 2, ..., N-1) * fg_mask (discard 0)
        # print('s, ', a_lbl.shape, self.n_classes, fg_mask.shape, F.one_hot(a_lbl.long()+1, num_classes=self.n_classes).shape)
        a_lbl = F.one_hot(a_lbl.long()+1, num_classes=self.n_classes).squeeze(0).permute(2, 0, 1) * fg_mask
        # a_lbl = F.one_hot(a_lbl.long()+1, num_classes=self.n_classes).permute(2, 0, 1) * fg_mask
        # initial a_lbl (BG, 1, 2, ..., N-1, 255) (255 for order 0)
        # a_lbl (255->0, 1, 2, ..., N-1) * BG_mask
        # angle (0, 1, 2, ...., N-1) one_hot, zero at BG pixels
        # become (BG, 0, 1, 2, ..., N) in model.py (1-> OOB pixels, ignore at training maybe)
        return a_lbl

    def __getitem__ (self, index):
        img_path = self.img_list[index]
        gt = self.gt_list[index]
        img = Image.open(img_path).convert('RGB')
        return img, gt, img_path
    
    def __len__(self):
        return len(self.img_list)
        
def draw_points(points, orders, size):
    maps = []
    for p, o in zip(points, orders):
        cntr = Image.new('L', size)
        draw = ImageDraw.Draw(cntr)
        if o == 0:
            o = 255
        draw.point(p, fill=o)
        cntr = np.asarray(cntr).astype(np.float32)
        maps.append(cntr)
    return maps

class NewSymmetryDatasets(NewSymmetryDatasetsBase):
    def __init__(self, sym_type='rotation', input_size=(417, 417), get_polygon=2, split='train', root='./sym_datasets/DENDI', \
                get_theta=False, n_classes=21, with_ref_circle=1, t_resize=False, n_theta=8):
        super(NewSymmetryDatasets, self).__init__(sym_type, get_polygon, split, root, with_ref_circle, n_theta)
        self.label = [sym_type]
        self.sym_type = sym_type
        self.split = split
        self.input_size = input_size
        self.get_theta = get_theta
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        if self.split == 'all':
            self.mean = [0, 0, 0]
            self.std = [1, 1, 1]
        self.n_classes = n_classes
        self.angle_interval = (360 // n_theta) #360/8=45
        self.n_theta = n_theta
        self.t_resize = t_resize #train:t_resize=False

    def process_data(self, gt, size):
        if self.sym_type in ['rotation']:
            return None, self.process_data_rot(gt, size)
        elif self.sym_type in ['reflection']:
            return self.process_data_ref(gt, size), None
        elif self.sym_type in ['joint']:
            return self.process_data_ref(gt, size), self.process_data_rot(gt, size)
        return gt

    def transform_data(self, img, gt, transform, reflection=True, t_resize=None):
        if gt is None:
            return None

        if reflection:#反射对称
            axis, axis_lbl, axis_coords = gt['axis'], gt['axis_lbl'], gt['line_coords']
            #axis为前景背景类，axis_lbl为对称轴得分图，axis_coords为x/宽和y/高
            
            axis_coords1, axis_coords2 = [], []
            for c in axis_coords:
                axis_coords1.append([c[0], c[1], c[0], c[1]])#第一个点的坐标
                axis_coords2.append([c[2], c[3], c[2], c[3]])#第二个点的坐标
            axis_gs = cv2.GaussianBlur(axis, (5,5), cv2.BORDER_DEFAULT)#高斯滤波去噪
            axis_gs = np.clip(axis_gs, 0, 0.21260943) # in case of the intersections
            # #截取数组中小于或者大于某值的部分,并使得被截取部分等于固定值 min:0, 0.21260943，max:0, 0.21260943
            # axis_gs是对前景背景类高斯模糊后得到的结果
        else:
            axis, a_lbl = gt['axis_map'], gt['order_map']
            axis_gs = cv2.GaussianBlur(axis, (11, 11), cv2.BORDER_DEFAULT)
            axis_gs = np.clip(axis_gs, 0, 0.01) # in case of the intersections

        if self.split in ['test', 'val', 'all'] and t_resize is not None:
            t_resize = t_resize(image=img, axis_gs=axis_gs)
            img, axis_gs = t_resize['image'], t_resize['axis_gs']

        if reflection:
            #反射对称，这一步得到图像增强后的图像，除此之外，这些标注也都进行同样的变化：
            # 前景背景类axis、前景背景类高斯模糊后axis_gs、对称得分图axis_lbl、
            # 对称轴第一个点的x/宽和y/高axis_coords1、对称轴第二个点的x/宽和y/高axis_coords2
            t = transform(image = img, axis = axis, axis_gs = axis_gs, axis_lbl=axis_lbl, axis_coords1=axis_coords1, axis_coords2=axis_coords2)
            img, axis, axis_gs, axis_lbl, axis_coords1, axis_coords2 = \
                t["image"], t["axis"], t["axis_gs"], t["axis_lbl"], t["axis_coords1"], t["axis_coords2"]
            axis_coords = []
            for a, b in zip(axis_coords1, axis_coords2):
                #使用 zip()函数压缩多个序列时，它会分别取各序列中第 1 个元素、... 第 n 个元素，各自组成新的元组
                axis_coords.append([a[0], a[1], b[0], b[1]])
        else:
            t = transform(image = img, axis = axis, axis_gs = axis_gs, a_lbl=a_lbl)
            img, axis, axis_gs, a_lbl = t["image"], t["axis"], t["axis_gs"], t["a_lbl"]

        mask = (axis_gs != 255).unsqueeze(0) #掩码mask，若不等于255则表示背景类为True，等于255则表示前景类为False
        #先判断每个值若不等于255则为True，等于255则为False。然后在0维度上插入一个维度，第0维度的大小默认是1
        axis = axis.unsqueeze(0)
        axis_gs = axis_gs.unsqueeze(0)
        axis_gs = axis_gs / (axis_gs.max() + 1e-5) #对高斯模糊后的前景背景类归一化
        r_dict = {'img': img, 'mask': mask, 'axis': axis, 'axis_gs': axis_gs}

        if reflection:
            r_dict['axis_lbl'], r_dict['axis_coords'] = axis_lbl, axis_coords
        else:
            r_dict['a_lbl'] = a_lbl
        return r_dict
                
    def __getitem__ (self, index):
        img = Image.open(self.img_list[index]).convert('RGB')
        ref_gt, rot_gt = self.process_data(self.gt_list[index], img.size) #返回reflection的gt_dict, None
        img = match_input_type(img) #转化为三通道图

        t_resize, rot_a_lbl, ref_a_lbl = None, 0, 0
        additional_targets={'axis': 'mask', 'axis_gs': 'mask', 'a_lbl': 'mask'}
        if self.split in ['test', 'val', 'all']:
            additional_targets['axis_lbl'] = 'mask'
            transform = A.Compose(
                        [ A.Normalize(self.mean, self.std),
                          ToTensorV2(),
                        ], additional_targets=additional_targets)#图像增强
            t_resize = A.Compose([
                A.LongestMaxSize(max_size=self.input_size[0]),
                A.PadIfNeeded(min_height=self.input_size[0], min_width=self.input_size[1], \
                              border_mode=cv2.BORDER_CONSTANT, mask_value=255),
            ], additional_targets={'axis_gs': 'mask'})
        else:#self.split为train, 执行这里
            additional_targets['axis_lbl'] = 'mask'
            additional_targets['axis_coords1'] = 'bboxes'
            additional_targets['axis_coords2'] = 'bboxes'

            # Python中的第三方图像增强库albumentations对图像进行数据增强的操作
            transform = A.Compose(
                    [ 
                        A.LongestMaxSize(max_size=self.input_size[0]),
                        A.PadIfNeeded(min_height=self.input_size[0], min_width=self.input_size[1], \
                                      border_mode=cv2.BORDER_CONSTANT,),
                    A.RandomRotate90(),
                    A.Rotate(limit = 15, border_mode = cv2.BORDER_CONSTANT),
                    A.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
                    A.Normalize(self.mean, self.std),
                    ToTensorV2(),
                    ], additional_targets=additional_targets)
            # A.LongestMaxSize：将图像的最大边缩放到指定的大小，这里的max_size为输入大小的第一个参数self.input_size[0]，这个函数会保持图像的长宽比。
            # A.PadIfNeeded：如果图像的长或宽小于指定的大小，则进行填充操作。这里的min_height和min_width分别为输入大小的第一个和第二个参数self.input_size[0]和self.input_size[1]，填充方式为BORDER_CONSTANT。
            # A.RandomRotate90：以90度为单位随机旋转图像。
            # A.Rotate：在一定角度范围内旋转图像，这里的限制为15度。
            # A.ColorJitter：随机调整图像的亮度、对比度、饱和度和色调。
            # A.Normalize：将图像像素值标准化为均值为self.mean，标准差为self.std的分布。ToTensorV2：将图像转换为张量。

        if not self.t_resize:#train的self.t_resize为False，执行后t_resize = None
            t_resize = None

        ref_return = self.transform_data(img, ref_gt, transform, True, t_resize)
        #返回经过数据增强处理过后的图像以及标注
        rot_return = self.transform_data(img, rot_gt, transform, False, t_resize)

        if self.get_theta:#为True进入
            if rot_gt is not None:
                rot_a_lbl = self.process_theta_rot(rot_return['a_lbl'])
            if ref_gt is not None:
                axis, axis_lbl, axis_coords = ref_return['axis'], ref_return['axis_lbl'], ref_return['axis_coords']
                if len(axis_coords) == 0:
                    ref_a_lbl = torch.zeros(self.n_theta, axis.shape[-2], axis.shape[-1])
                else:#存在对称轴坐标占比时
                    ref_a_lbl = self.process_theta_ref(axis_lbl, axis_coords).squeeze(0)
                    #ref_a_lbl

        if self.sym_type == 'reflection':#反射对称
            return ref_return['img'], ref_return['mask'], ref_return['axis'], ref_return['axis_gs'], False, ref_a_lbl
        elif self.sym_type == 'rotation':
            return rot_return['img'], rot_return['mask'], rot_return['axis'], rot_return['axis_gs'], False, rot_a_lbl
        elif self.sym_type == 'joint':
            ref_return = ref_return['img'], ref_return['mask'], ref_return['axis'], ref_return['axis_gs'], False, ref_a_lbl
            rot_return = rot_return['img'], rot_return['mask'], rot_return['axis'], rot_return['axis_gs'], False, rot_a_lbl
            return ref_return, rot_return

class CustomSymmetryDatasets(Dataset):#Dataset是代表这一数据的抽象类/基类
    def __init__(self, input_size=(417, 417), root='./demo/img'):
        super(CustomSymmetryDatasets, self).__init__()
        self.input_size = input_size #输入图形的大小
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.angle_interval = 45
        self.n_theta = 8
        self.img_list = self.get_img_list(root)

    def get_img_list(self, root_dir):#root_dir：数据目录
        img_names = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".png") or file.endswith(".jpg"):
                    img_names.append(os.path.join(root, file))
        return img_names
            
    def __getitem__ (self, index):
        img_path = self.img_list[index]
        img = Image.open(img_path).convert('RGB')
        img = match_input_type(img)

        transform = A.Compose([
                        A.Normalize(self.mean, self.std),
                        ToTensorV2(),
                        ])
        
        t_img = transform(image = img)["image"]
        return t_img, img_path

    def __len__(self):
        return len(self.img_list)
