import torch
import torchvision
import torch.nn as nn
import math
#os.environ['CUDA_VISIBLE_DEVICES']='0'
import time
from fp16util import network_to_half
#from apex import amp
root="/home/xu/py_work/flower102"



transforms=torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor()
        ])




data=torchvision.datasets.ImageFolder(root,transform=transforms)


BatchSize=16
#train_loder=torch.utils.data.DataLoader(data,shuffle=True,batch_size=BatchSize,num_workers=6,pin_memory=True)
torch.backends.cudnn.benchmark = True



class cnn_mnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv2d(3,32,3,1,1)
        self.c2=nn.Conv2d(32,64,3,1,1)
        self.c3=nn.Conv2d(64,64,3,1,1)
        self.c4=nn.Conv2d(64,64,3,1,1)
        self.maxpool=nn.MaxPool2d(2,2)

        
        
        
        
        self.fc1=nn.Linear(64*14*14,1000)
        self.fc2=nn.Linear(1000,102)
     
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
        out=self.fc2(out)
 
        return out
device=torch.device("cuda:0")
def get_mean(data):
    return sum(data)/len(data)       
model=cnn_mnist().to(device)
model=network_to_half(model)


criterion=nn.CrossEntropyLoss()
optimizers=torch.optim.Adam(model.parameters(),lr=1e-3)

#model, optimizer = amp.initialize(model, optimizers)

train_loder=torch.utils.data.DataLoader(data,shuffle=True,batch_size=BatchSize,num_workers=4,pin_memory=True)
from torch.optim.optimizer import Optimizer


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
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
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
optimizers= Adam16(model.parameters(),lr=1e-3)
for epoch in range(120):
  train_losses=[]
  train_acc=[]
  start=time.time()
 



  for step,(features,labels) in enumerate(train_loder):
#   if step==1:
    features,labels=features.to(device),labels.to(device)
    
    model.train()
    optimizers.zero_grad()
    result=model(features)
#    print(result.dtype)
#    for m in model.parameters():
#        print(m.dtype)
#    print(result.shape)
    
    acc=torch.sum(torch.eq(torch.max(result,1)[1].view_as(labels),labels)).item()/BatchSize
    train_acc.append(acc)

    loss=criterion(result,labels)
    train_losses.append(loss.item())
#    with amp.scale_loss(loss, optimizer) as scaled_loss:
#
#      scaled_loss.backward()
    loss.backward()
    optimizers.step()
    
    if step%100==0:
      print("epoch :",epoch,"acc ",get_mean(train_acc),"loss ",get_mean(train_losses)) 
  end=time.time()
  print(epoch,end-start)
#    val_losses=[]
#    val_acc=[]
#    for features,labels in val_loder:
#
#      features,labels=features.to(device),labels.to(device)
#      model.eval()
#      result=model.forward(features)
#      acc=torch.sum(torch.eq(torch.max(result,1)[1].view_as(labels),labels)).item()/Batch_Size
#      val_acc.append(acc)
#
#
#      loss=criterion(result,labels)
#      val_losses.append(loss.item())
#    print("epoch :",epoch,"acc ",get_mean(val_acc),"loss ",get_mean(val_losses)) 
    


