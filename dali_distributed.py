import torch
import torchvision
import os
import datetime
import time
from torch.optim.optimizer import Optimizer
import math
import torch.nn as nn


import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator 
import time
 
root="./imagenetPytorch"


class HybridTrainPipe(Pipeline):    
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False, local_rank=0, world_size=1):        
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)        
        dali_device = "gpu"        
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=True)        
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.RandomResizedCrop(device="gpu", size=crop, random_area=[0.08, 1.25])
       # self.res = ops.CenterCrop(device="gpu", size=crop)
         
        self.cmnp = ops.CropMirrorNormalize(device="gpu",output_dtype=types.FLOAT,output_layout=types.NCHW, 
        image_type=types.RGB,mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)        
        print('DALI "{0}" variant'.format(dali_device))     
    def define_graph(self):        
        rng = self.coin()        
        self.jpegs, self.labels = self.input(name="Reader")        
        images = self.decode(self.jpegs)        
        output = self.res(images)        
#        output = self.cmnp(images, mirror=rng)        
        return [output, self.labels] 

def get_imagenet_iter_dali(type, image_dir, batch_size, num_threads, device_id, num_gpus, crop, val_size=256,world_size=1,local_rank=0):
  
        pip_train = HybridTrainPipe(batch_size=batch_size, num_threads=num_threads, device_id=local_rank, data_dir=image_dir + '/train',  crop=crop, world_size=world_size, local_rank=local_rank)
        pip_train.build()        
        dali_iter_train = DALIClassificationIterator(pip_train, size=pip_train.epoch_size("Reader") // world_size)
        return dali_iter_train    



class cnn_mini(torch.nn.Module): 
     def __init__(self): 
         super().__init__() 
         self.c1=nn.Conv2d(3,32,3,1,1) 
         self.c2=nn.Conv2d(32,64,3,1,1) 
         self.c3=nn.Conv2d(64,64,3,1,1) 
         self.c4=nn.Conv2d(64,64,3,1,1) 
         self.maxpool=nn.MaxPool2d(2,2) 
 
 
          
          
          
          
         self.fc1=nn.Linear(64*14*14,1000) 
       
     def forward(self,x): 
         out=self.c1(x) 
         out=nn.functional.relu(out) 
         out=self.maxpool(out) 
         out=self.c2(out) 
         out=nn.functional.relu(out) 
         out=self.maxpool(out) 
         out=self.c3(out) 
         out=nn.functional.relu(out) 
         out=self.maxpool(out) 
         out=self.c4(out) 
         out=nn.functional.relu(out) 
         out=self.maxpool(out) 
         out=out.view(out.shape[0],-1) 
         out=self.fc1(out) 
    
   
         return out 



device=torch.device('cuda:0')
#model=network_to_half(model)
model=cnn_mini()
model=model.half()
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2,3,4,5,6,7])
model = model.to(device)
criterion=torch.nn.CrossEntropyLoss()


class Adam16(Optimizer): 


    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0): 
        defaults = dict(lr=lr, betas=betas, eps=eps, 
                        weight_decay=weight_decay) 
        params = list(params) 
        super(Adam16, self).__init__(params, defaults) 
        # for group in self.param_groups: 
            # for p in group['params']: 
         
        self.fp32_param_groups = [p.data.float().cuda() for p in params] 
        if not isinstance(self.fp32_param_groups[0], dict): 
            self.fp32_param_groups = [{'params': self.fp32_param_groups}] 


    def step(self, closure=None): 
        
         loss = None 
         if closure is not None: 
             loss = closure() 
 

         for group,fp32_group in zip(self.param_groups,self.fp32_param_groups): 
             for p,fp32_p in zip(group['params'],fp32_group['params']): 
                 if p.grad is None: 
                     continue 
                      
                 grad = p.grad.data.float() 
                 state = self.state[p] 
 

                 # State initialization 
                 if len(state) == 0: 
                     state['step'] = 0 
                     # Exponential moving average of gradient values 
                     state['exp_avg'] = grad.new().resize_as_(grad).zero_() 
                     # Exponential moving average of squared gradient values 
                     state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_() 
 

                 exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq'] 
                 beta1, beta2 = group['betas'] 
 

                 state['step'] += 1 
 

                 if group['weight_decay'] != 0: 
                     grad = grad.add(group['weight_decay'], fp32_p) 
 

                 # Decay the first and second moment running average coefficient 
                 exp_avg.mul_(beta1).add_(1 - beta1, grad) 
                 exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad) 
 

                 denom = exp_avg_sq.sqrt().add_(group['eps']) 
 

                 bias_correction1 = 1 - beta1 ** state['step'] 
                 bias_correction2 = 1 - beta2 ** state['step'] 
                 step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1 
              
                 # print(type(fp32_p)) 
                 fp32_p.addcdiv_(-step_size, exp_avg, denom) 
                 p.data = fp32_p.half() 
         return loss 





optimizers=Adam16(model.parameters(),lr=1e-3)

train_loader = get_imagenet_iter_dali(type='train', image_dir=root, batch_size=256*8, num_threads=4, crop=224, device_id=[0,1,2,3,4,5,6,7], num_gpus=8)
print('start iterate')
record=[]
for i, data in enumerate(train_loader):
    print(len(data))
    start=time.time()
    images = data[0]["data"].cuda(non_blocking=True).reshape(256*8,3,224,224).half()
    labels = data[0]["label"].squeeze().long().cuda(non_blocking=True)


    
    model.train()
#    result=torch.nn.parallel.data_parallel(model, images, device_ids=[0,1,2,3,4,5,6,7]).cuda()
      
            
    optimizers.zero_grad()
    result=model(images)
    loss=criterion(result,labels)
    loss.backward()
    optimizers.step()
    end = time.time()
#    print('end iterate')
    record.append((end-start))
    print(record)
#    print('dali iterate time: %fs' % (end - start))
#    print(result)
