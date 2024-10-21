import os
from glob import glob
import datetime
import numpy as np
import torch
import torch.utils.data as Data
from tqdm import tqdm
import kit
from utils import * 
import argparse

def train(args, model):
    if not os.path.exists(args.ckpt_pth):os.makedirs(args.ckpt_pth)
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
    
    files = np.array(glob(args.dataset, recursive=True))
    np.random.shuffle(files)
    files = files[:]
    points = kit.read_point_clouds(files)
    loader = Data.DataLoader(dataset = points, batch_size = args.batch_size, shuffle = True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    bpps, losses = [], []
    global_step = 0
    for epoch in range(1, 9999):
        printl(datetime.datetime.now())
        for step, (batch_x) in tqdm(enumerate(loader),total=len(loader)):

            B, N, _ = batch_x.shape  
            batch_x = batch_x.cuda()    

            optimizer.zero_grad()
            pred_batch_x, quantizated_feature, total_bits = model(batch_x) #bn3
            
            #loss
            bpp = total_bits / B / N
            mse = ((batch_x - pred_batch_x)**2).mean()
            loss = mse
            if global_step > args.rate_loss_enable_step:
                loss += args.bpp_lambda * bpp
            loss.backward()
            optimizer.step()
            global_step += 1

            losses.append(loss.item())
            bpps.append(bpp.item())
            if global_step % 500 == 0:
                printl(f'Epoch:{epoch} | Step:{global_step} | mse:{mse:.5f} | bpp:{round(np.array(bpps).mean(), 5)} | loss:{round(np.array(losses).mean(), 5)}')
                bpps, losses = [], []

            lr = args.lr
            if global_step % args.lr_decay_step == 0:
                lr = lr * args.weight_decay
                for g in optimizer.param_groups:
                    g['lr'] = lr
    
            if global_step % 500 == 0:
                torch.save(model.state_dict(), args.ckpt_dir + f'/{EXPERIMENT_NAME}.pt')
            if global_step >= args.max_steps:
                break
        if global_step >= args.max_steps:
            break
    return 

def parse_train_args():
    parser = argparse.ArgumentParser(description='Training Arguments')
    parser.add_argument('--dataset', type=str , default='/mnt/disk1/dataset/ShapeNet_pc_01_2048p_colorful/train/*.ply')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--downsample_rate',  type=float, default=4)
    parser.add_argument('--n_layer',  type=int, default=2)
    
    parser.add_argument('--bpp_lambda', default=3e-4 , type=float, help='bpp loss coefficient')
    parser.add_argument('--lr', type=float, default=5e-4, help='initial learning rate for backbone')
    parser.add_argument('--weight_decay', type=float, default= 0.1, help='weight decay for adam optimizer')
    parser.add_argument('--lr_decay_step', type=int, default=5000, help='learning rate decay step size')
    parser.add_argument('--max_steps', type=int, default=200000)

    parser.add_argument('--rate_loss_enable_step', type=int, default=5000, help='Apply rate-distortion tradeoff at x steps.')
    parser.add_argument('--log_dir', type=str, default='./Log', help='output path')
    parser.add_argument('--ckpt_dir', type=str, default='./model', help='checkpoint path')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    torch.cuda.manual_seed(1)
    torch.manual_seed(1)
    np.random.seed(1)
    
    args = parse_train_args()
    EXPERIMENT_NAME = f'net_{args.bpp_lambda}'
    LOG_PATH = './Log'+'/'+EXPERIMENT_NAME+'.txt'
    printl = CPrintl(LOG_PATH)
    
    from net import AE
    model = AE(n_layer=args.n_layer,ratio=args.downsample_rate).cuda().train()

    train(args, model)



