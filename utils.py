import os
import torch
import datetime
import numpy as np



class CPrintl():
    def __init__(self,logName) -> None:
        self.log_file = logName
        if os.path.dirname(logName)!='' and not os.path.exists(os.path.dirname(logName)):
            os.makedirs(os.path.dirname(logName))
    def __call__(self, *args):
        print(datetime.datetime.now().strftime('%Y-%m-%d:%H:%M:%S'),' ',*args)
        print(datetime.datetime.now().strftime('%Y-%m-%d:%H:%M:%S')+' ', *args, file=open(self.log_file, 'a'))
        
        

def pmf_to_cdf(pmf):
    cdf = pmf.cumsum(dim=-1)
    #print(cdf.shape)
    spatial_dimensions = pmf.shape[:-1] + (1,)
    zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
    cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
    # On GPU, softmax followed by cumsum can lead to the final value being 
    # slightly bigger than 1, so we clamp.
    cdf_with_0 = cdf_with_0.clamp(max=1.)
    return cdf_with_0


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


def RGB2YUV(rgb):
     
    m = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]])
     
    yuv = np.dot(rgb,m)
    yuv[:, 1:]+=128.0
    return yuv    


def YUV2RGB(yuv):
      
    m = np.array([[ 1.0, 1.0, 1.0],
                 [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                 [ 1.4019975662231445, -0.7141380310058594 , 0.00001542569043522235] ])
    
    rgb = np.dot(yuv,m)
    rgb[:,0]-=179.45477266423404
    rgb[:,1]+=135.45870971679688
    rgb[:,2]-=226.8183044444304
    return rgb


from plyfile import PlyData
def read_point_cloud(filepath):
    plydata = PlyData.read(filepath)
    try:
        pc = np.array(np.transpose(np.stack((plydata['vertex']['x'],plydata['vertex']['y'],plydata['vertex']['z'], plydata['vertex']['red'],plydata['vertex']['green'],plydata['vertex']['blue'])))).astype(np.float32)        
    except:
        pc = np.array(np.transpose(np.stack((plydata['vertex']['X'],plydata['vertex']['Y'],plydata['vertex']['Z'], plydata['vertex']['red'],plydata['vertex']['green'],plydata['vertex']['blue'])))).astype(np.float32)        
    pc[:, 3:] = RGB2YUV(pc[:, 3:]) / 255
    return pc


import multiprocessing
from tqdm import tqdm
def read_point_clouds(file_path_list):
    print('loading point clouds...')
    with multiprocessing.Pool(4) as p:
        pcs = list(tqdm(p.imap(read_point_cloud, file_path_list, 32), total=len(file_path_list)))
    return pcs



import pandas as pd
from pyntcloud import PyntCloud
def save_point_cloud(pc, path, save_color=False, save_normal=False):
    pc[:, 3:] = YUV2RGB(pc[:, 3:] * 255)
    pc[:, 3:] = np.round(np.clip(pc[:, 3:], 0, 255))
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




def divide(pc_path,save_path,points_num=8192):
    if not os.path.exists(save_path): os.makedirs(save_path)
    
    pc = read_point_cloud(pc_path)
    N,C = pc.shape
    set_num  = int(np.ceil(N/points_num)) 
    point_set = np.zeros((1,points_num,C)) 
    point_cloud = np.expand_dims(pc,0) 
    
    for i in range(set_num): 
        if i <set_num-1: 
            point_set = np.concatenate((point_set,point_cloud[:,i*points_num:(i+1)*points_num,:]),0) 
        else:  
            temp  = np.zeros((1,points_num,C)) 
            num_less_than_points_num = N-points_num*i 
            temp[:,0:num_less_than_points_num,:] = point_cloud[:,i*points_num:,:] 
            point_set = np.concatenate((point_set[1:,:,:],temp),0) 
    for i in tqdm(range(point_set.shape[0])): 
        path =  save_path + '/' + f'{i}.ply'
        pc_cube = point_set[i,:,:] 
        save_point_cloud(pc_cube, path, save_color=True)
    return






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

