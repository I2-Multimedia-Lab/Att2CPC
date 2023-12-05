import os
from glob import glob
import datetime
import numpy as np
import torch
import torch.utils.data as Data
from tqdm import tqdm
import kit

torch.cuda.manual_seed(1)
torch.manual_seed(1)
np.random.seed(1)

LAMBDA = 3e-4 
n_layer = 2
ratio = 4

EXPERIMENT_NAME = f'net_{LAMBDA}'
LOG_PATH = './Log'+'/'+EXPERIMENT_NAME+'.txt'
TRAIN_GLOB = '/mnt/disk1/dataset/ShapeNet_pc_01_2048p_colorful/train/*.ply'
MODEL_SAVE_FOLDER = f'./model'

BATCH_SIZE = 2
MAX_STEPS = 120000
LEARNING_RATE = 0.0005
LR_DECAY = 0.1
LR_DECAY_STEPS = 50000
RATE_LOSS_ENABLE_STEP = 5000

import datetime
class CPrintl():
    def __init__(self,logName) -> None:
        self.log_file = logName
        if os.path.dirname(logName)!='' and not os.path.exists(os.path.dirname(logName)):
            os.makedirs(os.path.dirname(logName))
    def __call__(self, *args):
        print(datetime.datetime.now().strftime('%Y-%m-%d:%H:%M:%S'),' ',*args)
        print(datetime.datetime.now().strftime('%Y-%m-%d:%H:%M:%S')+' ', *args, file=open(self.log_file, 'a'))
printl = CPrintl(LOG_PATH)

if not os.path.exists(MODEL_SAVE_FOLDER):os.makedirs(MODEL_SAVE_FOLDER)
if not os.path.exists('./Log'): os.makedirs('./Log')
    
files = np.array(glob(TRAIN_GLOB, recursive=True))
np.random.shuffle(files)
files = files[:]
points = kit.read_point_clouds(files)
loader = Data.DataLoader(dataset = points,batch_size = BATCH_SIZE,shuffle = True)

from net import AE
ae = AE(n_layer=n_layer,ratio=ratio).cuda().train()
optimizer = torch.optim.Adam(ae.parameters(), lr=LEARNING_RATE)

bpps, losses = [], []
global_step = 0
for epoch in range(1, 9999):
    printl(datetime.datetime.now())
    for step, (batch_x) in tqdm(enumerate(loader),total=len(loader)):

        B, N, _ = batch_x.shape  
        batch_x = batch_x.cuda()    

        optimizer.zero_grad()
        pred_batch_x, quantizated_feature, total_bits = ae(batch_x) #bn3
        
        #loss
        bpp = total_bits / B / N
        mse = ((batch_x - pred_batch_x)**2).mean()
        loss = mse
        if global_step > RATE_LOSS_ENABLE_STEP:loss += LAMBDA * bpp

        loss.backward()
        optimizer.step()
        global_step += 1

        losses.append(loss.item())
        bpps.append(bpp.item())
        if global_step % 500 == 0:
            printl(f'Epoch:{epoch} | Step:{global_step} | mse:{mse:.5f} | bpp:{round(np.array(bpps).mean(), 5)} | loss:{round(np.array(losses).mean(), 5)}')
            bpps, losses = [], []

        if global_step % LR_DECAY_STEPS == 0:
            LEARNING_RATE = LEARNING_RATE * LR_DECAY
            for g in optimizer.param_groups:
                g['lr'] = LEARNING_RATE
            printl(f'Learning rate decay triggered at step {global_step}, LR is setting to{LEARNING_RATE}.')
 
        if global_step % 500 == 0:
            torch.save(ae.state_dict(), MODEL_SAVE_FOLDER + '/' +  f'{EXPERIMENT_NAME}.pt')
        if global_step >= MAX_STEPS:
            break
    if global_step >= MAX_STEPS:
        break



