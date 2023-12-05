import os
from glob import glob
import numpy as np
from tqdm import tqdm
import torchac
import subprocess
import multiprocessing
import torch
from plyfile import PlyData
import pandas as pd
from pyntcloud import PyntCloud
import time
import shutil

# util fuctions
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



if __name__ == '__main__':
    
        
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    comp_f = f'./Compressed'
    decomp_f = f'./Decompressed'
    record_path = f'./baseline.txt'
    cubes_save_path = './temp_cubes'
    
    pc_name='Area_2'

    try:
        shutil.rmtree(cubes_save_path)
    except:
        pass
    for test_model in ['0.009','0.0003']:
        try:
            shutil.rmtree(comp_f)
            shutil.rmtree(decomp_f)
        except:
            pass

        pc_path = f'/mnt/ssd/lk/{pc_name}.ply'
        merge_path =f'./PredData/{pc_name}_rec.ply'
        model_save_path = f'./model/{test_model}.pt'
        
        if not os.path.exists(cubes_save_path): os.mkdir(cubes_save_path)
        if not os.path.exists(comp_f): os.mkdir(comp_f)
        if not os.path.exists(decomp_f): os.mkdir(decomp_f)
        if not os.path.exists(record_path): os.mknod(record_path)
        
        print('now loading model ...')                
        from net import AE 
        ae = AE(n_layer=2,ratio=4)
        ae.load_state_dict(torch.load(model_save_path))
        ae = ae.cuda().eval()
        cdf = pmf_to_cdf(ae.get_pmf('cuda')).cpu() 
        

        print('now start dividing ...')
        divide(pc_path,cubes_save_path,points_num=2048)
        cubes = './temp_cubes' + '/*.ply'
        pc_list = glob(cubes)

        print('now start test ...')
        with torch.no_grad():
            total_encode_time = 0
            total_decode_time = 0
            for f in tqdm(pc_list,total=len(pc_list)):

                pc = read_point_cloud(f) 
                pc = torch.Tensor(pc).cuda().unsqueeze(0) 
                
                fname = os.path.split(f)[-1]
                comp_name = os.path.join('/mnt/ssd/lk', fname + '.bin')
                decomp_name = os.path.join('/mnt/ssd/lk', fname + '.bin.ply')
                
                start_time= time.time()
                xyz,feature = pc[:,:,:3].transpose(1,2),pc[:,:,3:].transpose(1,2) 
                feature = ae.enc_emb(xyz,feature)
                xyz_ls = [xyz]
                for i in range(ae.n_layer):   
                    feature = ae.enc_dense_ls[i](xyz, feature) 
                    xyz, feature = ae.enc_ds_ls[i](xyz, feature)   
                    xyz_ls.append(xyz) 
                feature = ae.enc_comp(xyz, feature)
                
                quantizated_feature = torch.round(feature)
                quantizated_feature = quantizated_feature.transpose(1, 2).reshape(-1, 256) 
                quantizated_feature = quantizated_feature.to(torch.int16) + 99
                byte_stream = torchac.encode_float_cdf(cdf.repeat((quantizated_feature.shape[0], 1, 1)).cpu(), quantizated_feature.cpu(), check_input_bounds=True)
                
                with open(comp_name, 'wb') as fout:
                    fout.write(byte_stream)
                encode_patch_time = time.time() - start_time
                total_encode_time = total_encode_time + encode_patch_time
                
                quantizated_feature = torchac.decode_float_cdf(cdf.repeat((quantizated_feature.shape[0], 1, 1)).cpu(), byte_stream) 
                feature = quantizated_feature.float() - 99  
                feature = feature.unsqueeze(0).transpose(1,2).cuda()  
                
                start_time= time.time()
                feature = ae.dec_decomp(xyz, feature) 
                for i in range(ae.n_layer):
                    xyz, feature = ae.dec_us_ls[i](xyz, feature, xyz_ls[-2-i]) 
                    feature = ae.dec_dense_ls[i](xyz, feature) 
                feature = ae.dec_emb(xyz, feature).transpose(1,2)  
                pred_pc = torch.cat((xyz.transpose(1,2), feature), dim=-1)                    
                decode_patch_time = time.time() - start_time
                total_decode_time = total_decode_time + decode_patch_time
                save_point_cloud(pred_pc.squeeze(0).detach().cpu().numpy(), decomp_name, save_color=True)
        
        N= read_point_cloud(pc_path).shape[0]  
        files = np.array(glob('./Decompressed/*.ply', recursive=True))
        pc_ls = []   
        for i in range(len(files)):
            f = decomp_f +'/'+ f'{i}.ply.bin.ply'
            pc = read_point_cloud(f)
            pc_ls.append(pc)
        pc = np.vstack(pc_ls) 
        save_point_cloud(pc[:N,:], merge_path, save_color=True) 

        print('now start evaling ...')
        total_bits = 0
        filelist = glob(comp_f+'/*.bin', recursive=True)
        for f in filelist:
            bits =  os.stat(f).st_size * 8
            total_bits = total_bits + bits    
        color_bpp = total_bits / N    
        
        error_path = './pc_error.exe'
        cmd = f'wine ./pc_error.exe -a {merge_path} -b {pc_path} -c'
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)

        p2p_psnr = float(str(output).split('mseF,PSNR (p2point):')[1].split('\\r')[0])  #inf
        y_psnr = float(str(output).split('c[0],PSNRF         :')[1].split('\\r')[0])
        cb_psnr = float(str(output).split('c[1],PSNRF         :')[1].split('\\r')[0])
        cr_psnr = float(str(output).split('c[2],PSNRF         :')[1].split('\\r')[0])
        y_psnr, cb_psnr, cr_psnr = min(y_psnr, 100), min(cb_psnr, 100), min(cr_psnr, 100)
        yuv_psnr = (y_psnr + cb_psnr + cr_psnr)/3

        print(f'Name: {pc_name}\
            | Model: {test_model} \
            | p2pointPSNR: {p2p_psnr} \
            | bpp: {color_bpp} \
            | YPSNR: {y_psnr} \
            | CbPSNR: {cb_psnr} \
            | CrPSNR: {cr_psnr}\
            | YUVPSNR: {yuv_psnr}\
            | Encoding time: {total_encode_time}\
            | Decoding time: {total_decode_time} ')
        



                    