import torch
import torch.nn.functional as F
import torch.nn as nn 

from functions import reluPrime

class LossLayer(nn.Module): 
    
    def __init__(self, regularization_scale=0.0005, m_p=0.9, m_m=0.1, lmbd=0.5, cuda_enabled=True, device='cuda:0'): 
        super(LossLayer, self).__init__()
        
        self.cuda_enabled = cuda_enabled
        if cuda_enabled: 
            self.device = torch.device(device)
        else: 
            self.device = torch.device('cpu')
            
        self.regularization_scale = regularization_scale
        self.m_p = m_p
        self.m_m = m_m 
        self.lmbd = lmbd 
        
        self.zero = torch.zeros(1, device=self.device)
        
        
    def forward(self, out_digit, reconstruction, target, image): 
        
        bs = out_digit.size(0)
        
        self.v_c = torch.sqrt((out_digit**2).sum(dim=2))
        
        m1 = (torch.max(self.zero, self.m_p-self.v_c))**2
        m0 = (torch.max(self.zero, self.v_c-self.m_m))**2
        
        Lk = target * m1 + self.lmbd * (1.0 - target) * m0
        
        MarginL = Lk.sum(dim=1).mean(0)
        
        image = image.view(bs, -1)
        
        error = 2 *(reconstruction - image)**2
        ReconstructionL = error.sum(dim=1).mean(0)
        
        TotalL = MarginL + self.regularization_scale * ReconstructionL
        
        return TotalL, MarginL, ReconstructionL
        
    def backprop(self, out_digit, reconstruction, target, image): 
    
        bs = out_digit.size(0)
        image = image.view(bs, -1)
        
        dreconstruction = 4 * (reconstruction - image) * self.regularization_scale / bs
        
        dvc = -2. * torch.max(self.zero, self.m_p-self.v_c) * reluPrime(self.m_p-self.v_c) * target / bs
        dvc += 2. * torch.max(self.zero, self.v_c-self.m_m) * reluPrime(self.v_c-self.m_m) * self.lmbd * (1.-target) / bs
        
        dout_digit = dvc.unsqueeze(2) * out_digit / self.v_c.unsqueeze(2)
        
        return dreconstruction, dout_digit
        