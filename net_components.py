import os
import torch
import torch.nn as nn
import math
from pytorch3d.ops.knn import knn_gather, knn_points
from utils import *

class MLP(nn.Module):
    def __init__(self, in_channel, mlp, relu, bn):
        super(MLP, self).__init__()

        mlp.insert(0, in_channel)
        self.mlp_Modules = nn.ModuleList()
        for i in range(len(mlp) - 1):
            if relu[i]:
                if bn[i]:
                    mlp_Module = nn.Sequential(
                        nn.Conv2d(mlp[i], mlp[i+1], 1),
                        nn.BatchNorm2d(mlp[i+1]),
                        nn.ReLU(),
                    )
                else:
                    mlp_Module = nn.Sequential(
                        nn.Conv2d(mlp[i], mlp[i+1], 1),
                        nn.ReLU(),
                    )
            else:
                mlp_Module = nn.Sequential(
                    nn.Conv2d(mlp[i], mlp[i+1], 1),
                )
            self.mlp_Modules.append(mlp_Module)


    def forward(self, points, squeeze=False):
        """
        Input:
            points: input points position data, [B, C, N]
        Return:
            points: feature data, [B, D, N]
        """
        if squeeze:
            points = points.unsqueeze(-1) # [B, C, N, 1]
        
        for m in self.mlp_Modules:
            points = m(points)
        # [B, D, N, 1]
        
        if squeeze:
            points = points.squeeze(-1) # [B, D, N] 

        return points

 


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
        
        pe = self.relu(torch.matmul(score, value)) # B, M, K, Cout
        # pe = pe.mean(dim=2) # B, M, Cout
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

        query = self.q_mlp(grouped_xyz).permute((0, 3, 2, 1)) 
        key = self.k_mlp(grouped_feature).permute((0, 3, 2, 1))  
        value = self.v_mlp(grouped_feature).permute((0, 3, 2, 1))  

        relation_qk = key - query   
        pem = self.linear_p_multiplier(grouped_xyz)   
        relation_qk = relation_qk * pem  

        peb = self.linear_p_bias(grouped_xyz)  
        relation_qk = relation_qk + peb   
        value = value + peb  

        weight  = self.weight_encoding(relation_qk)
        score = self.softmax(weight)  
        feature = (score*value).sum(dim=2)  
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
        
        B, _, N = xyz.shape
        xyz, feature = xyz.transpose(1, 2), feature.transpose(1, 2)


        _, idx, grouped_xyz = knn_points(xyz, xyz, K=self.kernel_size, return_nn=True)  

        grouped_xyz = grouped_xyz - xyz.view(B, N, 1, 3)  
        grouped_xyz = n_scale_ball(grouped_xyz).transpose(1, 3) 

        if self.in_channel != self.out_channel:
            feature = self.linear_in(feature)
 
        for i in range(self.n_layers):   
            
            identity = feature
            feature = self.sa_emb_ls[i](feature) #

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
        B, _, N = xyz.shape
        xyz, feature = xyz.transpose(1, 2), feature.transpose(1, 2)

        M = N//self.ratio
        sampled_idx = farthest_point_sample_batch(xyz.cpu(), M).cuda()
        sampled, sampled_feature = index_points(xyz, sampled_idx), index_points(feature, sampled_idx)

        sampled, sampled_feature = sampled.transpose(1, 2), sampled_feature.transpose(1, 2)
        return sampled, sampled_feature


class UpSampling_KNN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, n_layers):
        super(UpSampling_KNN, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        
        self.sa_ls, self.sa_emb_ls = nn.ModuleList(), nn.ModuleList()
        for i in range(n_layers):
            self.sa_emb_ls.append(nn.Sequential(
                nn.Linear(out_channel, out_channel),
                nn.ReLU(),
            ))
            self.sa_ls.append(CrossAttention(out_channel))

    def forward(self, xyz, feature, xyz_dense):
        B, _, N = xyz.shape
        xyz, feature, xyz_dense = xyz.transpose(1, 2), feature.transpose(1, 2), xyz_dense.transpose(1, 2)

        _, idx, grouped_xyz = knn_points(xyz_dense, xyz, K=self.kernel_size, return_nn=True)
        grouped_xyz = grouped_xyz - xyz_dense.view(B, -1, 1, 3)
        grouped_xyz = n_scale_ball(grouped_xyz).transpose(1, 3)

        for i in range(self.n_layers):
            feature = self.sa_emb_ls[i](feature)
            grouped_feature = knn_gather(feature, idx).transpose(1, 3) # grouped_feature B, Cmid, K, N
            output = self.sa_ls[i](grouped_xyz, grouped_feature)
            feature = output

        xyz_dense, feature = xyz_dense.transpose(1, 2), feature.transpose(1, 2)
        return xyz_dense, feature
    

class UpSampling_Padzero(nn.Module):  
    def __init__(self):
        super(UpSampling_Padzero, self).__init__()

    def forward(self, xyz, feature, xyz_anchor):
        B, C, N = feature.shape

        xyz, feature, xyz_anchor = xyz.transpose(1, 2), feature.transpose(1, 2), xyz_anchor.transpose(1, 2)
        
        _, idx, grouped_xyz = knn_points(xyz_anchor, xyz, K=1, return_nn=True) 

        grouped_feature = knn_gather(feature, idx).squeeze(-2)
        grouped_xyz = grouped_xyz.view(B, -1, 3)

        feature_anchor = torch.zeros_like(grouped_feature).cuda()
        
        feature_anchor = torch.where(((grouped_xyz == xyz_anchor).sum(dim=-1) == 3).unsqueeze(-1).repeat((1, 1, C)),
                    grouped_feature, feature_anchor)
        
        xyz_anchor, feature_anchor = xyz_anchor.transpose(1, 2), feature_anchor.transpose(1, 2)

        return xyz_anchor, feature_anchor
    
