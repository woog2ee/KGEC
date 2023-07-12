import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from model.binary_classification import BERTBinaryClassification



class BERTTrainer:
    def __init__(self, bert, lr, betas, weight_decay, warmup_steps):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.bert  = bert
        self.model = BERTBinaryClassification(bert).to(self.device)
        self.criterion = nn.BCELoss()
        
        self.optim = Adam(params       = self.model.parameters(),
                          lr           = lr,
                          betas        = betas,
                          weight_decay = weight_decay)
        self.optim_schedule = ScheduledOptim(optimizer    = self.optim,
                                             d_model      = self.bert.d_model,
                                             warmup_steps = warmup_steps)
        
        
    def do(self, mode, epoch, data_loader):
        if mode == 'train':
            return self.iteration(epoch, data_loader)
        elif mode == 'test':
            return self.iteration(epoch, data_loader, train=False)
        
    
    def iteration(self, epoch, data_loader, train=True):
        if train:
            mode = 'train'
            self.model.train()
        else:
            mode = 'test'
            self.model.eval()
        
        data_iter = tqdm(enumerate(data_loader),
                         desc='EP_%s:%d' % (mode, epoch),
                         total=len(data_loader),
                         bar_format='{l_bar}{r_bar}')
        
        # Iteration
        avg_loss = 0.0
        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}
            
            output = self.model.forward(data['input'])
            output = torch.tensor(torch.argmax(output, dim=2), dtype=torch.float16)
            label  = torch.tensor(data['label'], dtype=torch.float16)
                                  
            loss = self.criterion(output, label)
            avg_loss += loss.item()

            if train:
                self.optim_schedule.zero_grad()
                loss.requires_grad_(True)
                loss.backward()
                self.optim_schedule.step_and_update_lr()
                    
            post_fix = {'epoch'   : epoch,
                        'iter'    : i,
                        'avg_loss': avg_loss / (i+1),
                        'loss'    : loss.item()}
            if i%100 == 0: data_iter.write(str(post_fix))

        return avg_loss
       
    
    def save(self, epoch, output_path):
        output_path = output_path + f'berttrained_epoch{epoch}.pt'
        #torch.save(self.model.cpu(), output_path)
        torch.save(self.model.state_dict(), output_path)
        self.model.to(self.device)
        print(f'Epoch {epoch} Model Saved at {output_path}\n')
        return output_path
    
    
    
class ScheduledOptim():
    def __init__(self, optimizer, d_model, warmup_steps):
        self._optimizer   = optimizer
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.init_lr      = np.power(d_model, -0.5)
        
        
    def step_and_update_lr(self):
        self._update_lr()
        self._optimizer.step()
        
        
    def zero_grad(self):
        self._optimizer.zero_grad()
        
        
    def _get_lr_scale(self):
        return np.min([
            np.power(self.current_step, -0.5),
            np.power(self.warmup_steps, -1.5) * self.current_step
        ])
    
    
    def _update_lr(self):
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale()
        
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr