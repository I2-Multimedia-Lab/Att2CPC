import math
import torch
import torch.nn as nn
from net_components import PointConv, DownSampling, UpSampling_Padzero, BitEstimator

class AE(nn.Module):  
    def __init__(self, n_layer,ratio): 
        super(AE, self).__init__()

        self.enc_emb = PointConv(in_channel=3, out_channel=256, kernel_size=8, n_layers=2)
        self.enc_dense_ls, self.enc_ds_ls = nn.ModuleList(),nn.ModuleList()
        for i in range(n_layer):
            self.enc_dense_ls.append(PointConv(in_channel=256, out_channel=256, kernel_size=8, n_layers=4))
            self.enc_ds_ls.append(DownSampling(ratio=ratio))
        self.enc_comp = PointConv(in_channel=256, out_channel=256, kernel_size=8, n_layers=2)
        self.dec_decomp = PointConv(in_channel=256, out_channel=256, kernel_size=8, n_layers=2)
        self.dec_dense_ls, self.dec_us_ls = nn.ModuleList(),nn.ModuleList()
        for i in range(n_layer):
            self.dec_us_ls.append(UpSampling_Padzero())
            self.dec_dense_ls.append(PointConv(in_channel=256, out_channel=256, kernel_size=8, n_layers=4))
        self.dec_emb = PointConv(in_channel=256, out_channel=3, kernel_size=8, n_layers=2)
        self.be = BitEstimator(channel=256)
        self.n_layer = n_layer
    
    def forward(self, batch_x):
        
        #Encoder
        xyz, feature = batch_x[:, :, :3], batch_x[:, :, 3:]
        xyz, feature = xyz.transpose(1, 2), feature.transpose(1, 2)  
        feature = self.enc_emb(xyz, feature) 
        xyz_ls = [xyz]
        for i in range(self.n_layer):   
            feature = self.enc_dense_ls[i](xyz, feature) 
            xyz, feature = self.enc_ds_ls[i](xyz, feature)   
            xyz_ls.append(xyz) 
        feature = self.enc_comp(xyz, feature)
        
        #Quantize 
        if self.training:
            quantizated_feature = feature + torch.nn.init.uniform_(torch.zeros(feature.size()), -0.5, 0.5).cuda()
        else:
            quantizated_feature = torch.round(feature) 
        feature = quantizated_feature
        
        #Decder
        feature = self.dec_decomp(xyz, feature)
        for i in range(self.n_layer):
            xyz, feature = self.dec_us_ls[i](xyz, feature, xyz_ls[-2-i])
            feature = self.dec_dense_ls[i](xyz, feature) 
        feature = self.dec_emb(xyz, feature) 
        
        #Get output
        quantizated_feature = quantizated_feature.transpose(1, 2).reshape(-1, 256) 
        prob = self.be(quantizated_feature + 0.5) - self.be(quantizated_feature - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
        xyz, feature = xyz.transpose(1, 2), feature.transpose(1, 2)
        new_batch_x = torch.cat((xyz, feature), dim=-1)
        
        return new_batch_x, quantizated_feature, total_bits


    def get_pmf(self, device='cuda'): 
        self.d = 256
        L = 99  
        pmf = torch.zeros(1, self.d, L*2).to(device)
        for l in range(-L, L):
            z = torch.ones((1, self.d)).to(device) * l
            pmf[0, :, l+L] = (self.be(z + 0.5) - self.be(z - 0.5))[0, :]
        return pmf

