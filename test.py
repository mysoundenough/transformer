import torch
import torch.nn.functional as F

a = [[-1,0,0.5,0.5], [-1,-1,0.5,0.5]]
data = torch.tensor(a)
# data=torch.randn(2,5)
print(data)

softmax= F.softmax(data,dim=-1)
print(softmax)

log_soft1=torch.log(softmax)
print(log_soft1)