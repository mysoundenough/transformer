import torch
import torch.nn.functional as F
import torch.nn as nn

""" 
# softmax 与 logsoftmax
a = [[-1,0,0.5,0.5], [-1,-1,0.5,0.5]]
data = torch.tensor(a)
# data=torch.randn(2,5)
print(data)

softmax= F.softmax(data,dim=-1)
print(softmax)

log_soft1=torch.log(softmax)
print(log_soft1)
"""


batch_size = 5
nb_classes = 2
in_features = 10

model = nn.Linear(in_features, nb_classes)
criterion = nn.CrossEntropyLoss()

x = torch.randn(batch_size, in_features)
target = torch.empty(batch_size, dtype=torch.long).random_(nb_classes)
print(target)

output = model(x)
loss = criterion(output, target)
loss.backward()