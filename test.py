import os
from glob import glob
import numpy as np
from tqdm import tqdm
import torchac
import subprocess
import torch
import time
import shutil

from utils import *


import argparse
def parse_test_args():
    parser = argparse.ArgumentParser(description='Training Arguments')
    parser.add_argument('--compressed_dir', type=str , default='./Compressed')
    parser.add_argument('--decompressed_dir', type=str , default='./Decompressed')
    parser.add_argument('--record_pth', type=str , default='./record.txt')
    parser.add_argument('--cubes_dir', type=str , default='./temp_cubes')
    parser.add_argument('--merge_dir', type=str , default='./PredData')
    parser.add_argument('--test_pc_dir', type=str , default='/mnt/ssd/test_pc')
    parser.add_argument('--model_save_dir', type=str , default='./model')
    parser.add_argument('--pc_error_pth', type=str , default='./pc_error.exe')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = parse_test_args()
    printl = CPrintl(args.record_pth)
    
    
    pc_name='Area_2'
    pc_path = f'/mnt/ssd/lk/{pc_name}.ply'
    merge_path =f'./PredData/{pc_name}_rec.ply'

    model_list = os.listdir(args.model_save_dir)
    for test_model in model_list:
        shutil.rmtree(args.cubes_dir)
        shutil.rmtree(args.compressed_dir)
        shutil.rmtree(args.decompressed_dir)
      
        model_save_path = f'./model/{test_model}.pt'

        if not os.path.exists(args.cubes_dir): os.mkdir(args.cubes_dir)
        if not os.path.exists(args.compressed_dir): os.mkdir(args.compressed_dir)
        if not os.path.exists(args.decompressed_dir): os.mkdir(args.decompressed_dir)
        if not os.path.exists(args.record_pth): os.mknod(args.record_pth)
        
        #load pretrain model
        from net import AE 
        ae = AE(n_layer=2,ratio=4)
        ae.load_state_dict(torch.load(model_save_path))
        ae = ae.cuda().eval()
        cdf = pmf_to_cdf(ae.get_pmf('cuda')).cpu() 

        #divide into cubes
        divide(pc_path,args.cubes_dir,points_num=2048)
        cube_list = glob('./temp_cubes/*.ply')

        with torch.no_grad():
            total_encode_time = 0
            total_decode_time = 0
            for f in tqdm(cube_list,total=len(cube_list)):

                pc = read_point_cloud(f) 
                pc = torch.Tensor(pc).cuda().unsqueeze(0) 
                
                fname = os.path.split(f)[-1]
                comp_name = args.compressed_dir + f'{fname}.bin'
                decomp_name = args.decompressed_dir + f'{fname}.bin.ply'
                
                #compress
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
                
                
                #decompress
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
        
        
        #merge cubes
        N= read_point_cloud(pc_path).shape[0]  
        files = np.array(glob('./Decompressed/*.ply', recursive=True))
        pc_ls = []   
        for i in range(len(files)):
            f = args.decompressed_dir +f'/{i}.ply.bin.ply'
            pc = read_point_cloud(f)
            pc_ls.append(pc)
        pc = np.vstack(pc_ls) 
        save_point_cloud(pc[:N,:], merge_path, save_color=True) 


        #cal bpp
        total_bits = 0
        filelist = glob(args.compressed_dir+'/*.bin', recursive=True)
        for f in filelist:
            bits =  os.stat(f).st_size * 8
            total_bits = total_bits + bits    
        color_bpp = total_bits / N    
        
        #cal psnr
        cmd = f'wine {args.pc_error_pth} -a {merge_path} -b {pc_path} -c'
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)

        p2p_psnr = float(str(output).split('mseF,PSNR (p2point):')[1].split('\\r')[0])  #inf
        y_psnr = float(str(output).split('c[0],PSNRF         :')[1].split('\\r')[0])
        cb_psnr = float(str(output).split('c[1],PSNRF         :')[1].split('\\r')[0])
        cr_psnr = float(str(output).split('c[2],PSNRF         :')[1].split('\\r')[0])
        y_psnr, cb_psnr, cr_psnr = min(y_psnr, 100), min(cb_psnr, 100), min(cr_psnr, 100)
        yuv_psnr = (y_psnr + cb_psnr + cr_psnr)/3

        printl(f'Name: {pc_name}\
            | Model: {test_model} \
            | p2pointPSNR: {p2p_psnr} \
            | bpp: {color_bpp} \
            | YPSNR: {y_psnr} \
            | CbPSNR: {cb_psnr} \
            | CrPSNR: {cr_psnr}\
            | YUVPSNR: {yuv_psnr}\
            | Encoding time: {total_encode_time}\
            | Decoding time: {total_decode_time} ')
        




                    
                    