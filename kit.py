import os
import multiprocessing
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import math
from plyfile import PlyData
import pandas as pd
from pyntcloud import PyntCloud
from pytorch3d.ops.knn import knn_gather, knn_points


def normalize(pc, margin=0.01):
    # pc: (1, N, 3), one point cloud
    # margin: rescaling pc to [0+margin, 1-margin]
    # device = pc.device

    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
    center = np.array([(x.max()+x.min())/2, (y.max()+y.min())/2, (z.max()+z.min())/2])
    longest = np.max(np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]))

    pc = pc - center
    pc = pc * (1-margin) / longest
    pc = pc + 0.5
    
    return pc

#input is a RGB numpy array with shape (height,width,3), can be uint,int, float or double, values expected in the range 0..255
#output is a double YUV numpy array with shape (height,width,3), values in the range 0..255
def RGB2YUV(rgb):
     
    m = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]])
     
    yuv = np.dot(rgb,m)
    yuv[:, 1:]+=128.0
    return yuv

#input is an YUV numpy array with shape (height,width,3) can be uint,int, float or double,  values expected in the range 0..255
#output is a double RGB numpy array with shape (height,width,3), values in the range 0..255
def YUV2RGB(yuv):
      
    m = np.array([[ 1.0, 1.0, 1.0],
                 [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                 [ 1.4019975662231445, -0.7141380310058594 , 0.00001542569043522235] ])
    
    rgb = np.dot(yuv,m)
    rgb[:,0]-=179.45477266423404
    rgb[:,1]+=135.45870971679688
    rgb[:,2]-=226.8183044444304
    return rgb


def read_point_cloud(filepath):
    plydata = PlyData.read(filepath)
    try:
        pc = np.array(np.transpose(np.stack((plydata['vertex']['x'],plydata['vertex']['y'],plydata['vertex']['z'], plydata['vertex']['red'],plydata['vertex']['green'],plydata['vertex']['blue'])))).astype(np.float32)        
    except:
        pc = np.array(np.transpose(np.stack((plydata['vertex']['X'],plydata['vertex']['Y'],plydata['vertex']['Z'], plydata['vertex']['red'],plydata['vertex']['green'],plydata['vertex']['blue'])))).astype(np.float32)        
    pc[:, 3:] = RGB2YUV(pc[:, 3:]) / 255
    return pc


def read_point_clouds(file_path_list):
    print('loading point clouds...')
    with multiprocessing.Pool(4) as p:
        pcs = list(tqdm(p.imap(read_point_cloud, file_path_list, 32), total=len(file_path_list)))
    return pcs


def save_point_cloud(pc, path, save_color=False, save_normal=False):
    pc[:, 3:] = YUV2RGB(pc[:, 3:] * 255)
    if save_color and save_normal:
        pc = pd.DataFrame(pc, columns=['x', 'y', 'z', 'red', 'green', 'blue', 'nx', 'ny', 'nz'])
        pc.fillna(0, inplace=True)
        pc[['red','green','blue']] = np.round(pc[['red','green','blue']]).astype(np.uint8)
    elif save_color:
        pc = pd.DataFrame(pc, columns=['x', 'y', 'z', 'red', 'green', 'blue'])
        pc.fillna(0, inplace=True)
        pc[['red','green','blue']] = np.round(np.clip(pc[['red','green','blue']], 0, 255)).astype(np.uint8)
    elif save_normal:
        pc = pd.DataFrame(pc, columns=['x', 'y', 'z', 'nx', 'ny', 'nz'])
    else:
        pc = pd.DataFrame(pc, columns=['x', 'y', 'z'])
    cloud = PyntCloud(pc)
    cloud.to_file(path)


def farthest_point_sample_batch(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S] or [B, S, K]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    batch_indices = batch_indices.view(view_shape)
    batch_indices = batch_indices.repeat(repeat_shape)
    new_points = points[batch_indices, idx.long(), :]
    return new_points

def n_scale_ball(grouped_xyz):  
    B, N, K, _ = grouped_xyz.shape
    longest = (grouped_xyz**2).sum(dim=-1).sqrt().max(dim=-1)[0]
    scaling = (1) / longest   
    grouped_xyz = grouped_xyz * scaling.view(B, N, 1, 1)
    return grouped_xyz


# class CrossAttention(nn.Module):
#     def __init__(self, channel):
#         super(CrossAttention, self).__init__()
#         self.channel = channel
#         self.q_mlp = nn.Conv2d(in_channels=3, out_channels=channel, kernel_size=1)
#         self.k_mlp = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
#         self.v_mlp = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
#         self.softmax = nn.Softmax(dim=-1)
#         self.relu = nn.ReLU()

#     def forward(self, grouped_xyz, grouped_feature):
#         '''
#         input grouped_feature: B, Cin, K, M
#         output grouped_feature: B, Cout, K, M
#         '''
#         query = self.q_mlp(grouped_xyz).permute((0, 3, 2, 1)) # B, M, K, Cout
#         key = self.k_mlp(grouped_feature).permute((0, 3, 2, 1)) # B, M, K, Cout
#         value = self.v_mlp(grouped_feature).permute((0, 3, 2, 1)) # B, M, K, Cout
#         score = torch.matmul(query, key.transpose(2, 3)) # B, M, K, K
#         score = self.softmax(score/math.sqrt(self.channel)) # B, M, K, K
        
#         grouped_feature = self.relu(torch.matmul(score, value)) # B, M, K, Cout
#         grouped_feature = grouped_feature.mean(dim=2) # B, M, Cout
#         return grouped_feature




class SelfAttention(nn.Module):
    def __init__(self, channel):
        super(SelfAttention, self).__init__()
        self.channel = channel
        self.q_mlp = nn.Conv2d(in_channels=3, out_channels=channel, kernel_size=1)
        self.k_mlp = nn.Conv2d(in_channels=3, out_channels=channel, kernel_size=1)
        self.v_mlp = nn.Conv2d(in_channels=3, out_channels=channel, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, grouped_xyz):
        '''
        input grouped_feature: B, Cin, K, M
        output grouped_feature: B, Cout, K, M
        '''
        query = self.q_mlp(grouped_xyz).permute((0, 3, 2, 1)) # B, M, K, Cout
        key = self.k_mlp(grouped_xyz).permute((0, 3, 2, 1)) # B, M, K, Cout
        value = self.v_mlp(grouped_xyz).permute((0, 3, 2, 1)) # B, M, K, Cout
        score = torch.matmul(query, key.transpose(2, 3)) # B, M, K, K
        score = self.softmax(score/math.sqrt(self.channel)) # B, M, K, K
        
        pe = self.relu(torch.matmul(score, value)) 

        return pe




class CrossAttention(nn.Module):
    def __init__(self, channel):
        super(CrossAttention, self).__init__()
        self.channel = channel
        self.q_mlp = nn.Conv2d(in_channels=3, out_channels=channel, kernel_size=1)
        self.k_mlp = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        self.v_mlp = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
 
        self.linear_p_multiplier = SelfAttention(channel)
        self.linear_p_bias = SelfAttention(channel)
        
        self.weight_encoding = nn.Sequential(nn.Linear(channel, channel),nn.ReLU(inplace=True),nn.Linear(channel, channel))
        self.residual_emb = nn.Sequential(nn.ReLU(),nn.Linear(channel, channel))
        self.softmax = nn.Softmax(dim=2)

    def forward(self, grouped_xyz, grouped_feature):
        '''
        input grouped_feature: B, Cin, K, M
        output grouped_feature: B, Cout, K, M
        '''
        query = self.q_mlp(grouped_xyz).permute((0, 3, 2, 1)) # B, M, K, Cout
        key = self.k_mlp(grouped_feature).permute((0, 3, 2, 1)) # B, M, K, Cout
        value = self.v_mlp(grouped_feature).permute((0, 3, 2, 1)) # B, M, K, Cout

        relation_qk = key - query  
        pem = self.linear_p_multiplier(grouped_xyz)  
        relation_qk = relation_qk * pem  

        peb = self.linear_p_bias(grouped_xyz) #bskc
        relation_qk = relation_qk + peb  #bskc
        value = value + peb  #bskc

        weight  = self.weight_encoding(relation_qk)
        score = self.softmax(weight) # B, M, K, Cout
        feature = (score*value).sum(dim=2) # B, M, Cout
        feature = self.residual_emb(feature)

        return feature



class PointConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, n_layers):
        super(PointConv, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.n_layers = n_layers


        self.sa_ls, self.sa_emb_ls = nn.ModuleList(), nn.ModuleList()

        if self.in_channel != self.out_channel:
            self.linear_in = nn.Linear(in_channel, out_channel)
        for i in range(n_layers):                        
            self.sa_emb_ls.append(nn.Sequential(    
                nn.Linear(out_channel, out_channel),
                nn.ReLU(),
            ))
            self.sa_ls.append(CrossAttention(out_channel))  
    def forward(self, xyz, feature): 
        """
        Input:
            xyz: input points position data, [B, 3, N]
            feature: input points feature data, [B, Cin, N]
        Return:
            feature: output feature data, [B, Cout, N]
        """
        B, _, N = xyz.shape
        xyz, feature = xyz.transpose(1, 2), feature.transpose(1, 2)
        _, idx, grouped_xyz = knn_points(xyz, xyz, K=self.kernel_size, return_nn=True) 
        grouped_xyz = grouped_xyz - xyz.view(B, N, 1, 3)  
        grouped_xyz = n_scale_ball(grouped_xyz).transpose(1, 3)

        if self.in_channel != self.out_channel:
            feature = self.linear_in(feature)
 
        for i in range(self.n_layers):             
            identity = feature
            feature = self.sa_emb_ls[i](feature) 
            grouped_feature = knn_gather(feature, idx).transpose(1, 3) 
            output = self.sa_ls[i](grouped_xyz, grouped_feature) 
            feature = identity + output  
        feature = feature.transpose(1, 2)

        return feature


class DownSampling(nn.Module):
    def __init__(self, ratio):
        super(DownSampling, self).__init__()
        self.ratio = ratio

    def forward(self, xyz, feature):
        """
        Input:
            xyz: input points position data, [B, 3, N]
            feature: input points feature data, [B, Cin, N]
        Return:
            feature: output feature data, [B, Cout, N]
        """
        B, _, N = xyz.shape
        xyz, feature = xyz.transpose(1, 2), feature.transpose(1, 2)

        M = N//self.ratio
        sampled_idx = farthest_point_sample_batch(xyz, M)
        sampled, sampled_feature = index_points(xyz, sampled_idx), index_points(feature, sampled_idx)

        sampled, sampled_feature = sampled.transpose(1, 2), sampled_feature.transpose(1, 2)
        return sampled, sampled_feature


class UpSampling_Padzero(nn.Module): 
    def __init__(self):
        super(UpSampling_Padzero, self).__init__()

    def forward(self, xyz, feature, xyz_anchor):
        """
        Input:
            xyz: input points position data, [B, 3, N]
            feature: input points feature data, [B, Cin, N]
        Return:
            feature: output feature data, [B, Cout, N]
        """
        B, C, N = feature.shape
        xyz, feature, xyz_anchor = xyz.transpose(1, 2), feature.transpose(1, 2), xyz_anchor.transpose(1, 2)
        _, idx, grouped_xyz = knn_points(xyz_anchor, xyz, K=1, return_nn=True) 
        grouped_feature = knn_gather(feature, idx).squeeze(-2)
        grouped_xyz = grouped_xyz.view(B, -1, 3)
        feature_anchor = torch.zeros_like(grouped_feature).cuda()
        feature_anchor = torch.where(((grouped_xyz == xyz_anchor).sum(dim=-1) == 3).unsqueeze(-1).repeat((1, 1, C)),grouped_feature, feature_anchor)
        xyz_anchor, feature_anchor = xyz_anchor.transpose(1, 2), feature_anchor.transpose(1, 2)

        return xyz_anchor, feature_anchor
    

class SelfChannelAttention(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SelfChannelAttention, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.q_mlp = nn.Conv1d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=1)
        self.k_mlp = nn.Conv1d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=1)
        self.v_mlp = nn.Conv1d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feature):
        """
        Input:
            feature: input points feature data, [B, Cin, N]
        Return:
            feature: output feature data, [B, Cout, N]
        """
        query = self.q_mlp(feature) # B, Cout, N
        key = self.k_mlp(feature) # B, Cout, N
        value = self.v_mlp(feature) # B, Cout, N
        score = torch.matmul(query, key.transpose(1, 2)) # B, Cout, Cout
        score = self.softmax(score/math.sqrt(self.out_channel)) # B, Cout, Cout
        
        feature = torch.matmul(score, value) # B, Cout, N
        return feature

