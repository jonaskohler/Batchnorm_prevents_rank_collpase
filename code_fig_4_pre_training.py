#################################################################
######################## CODE FIGURE 4 ##########################
#################################################################


# Run with passing number of hidden layers as argument


import numpy as np
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import sys
import torch.nn.functional as func
import pickle
import sys, getopt
import IPython
import math

dtype = torch.cuda.FloatTensor 
dtype_labels = torch.cuda.LongTensor 
no_of_hl= int(sys.argv[1])   # number of hidden layers
HUs=128 # number of hidden units
step_size = 0.01 # stepsize
min_batch_size = 64 # minibatch size for SGD
batch_norm_size = 10
hidden_layers=np.ones(no_of_hl,dtype=int)*HUs  


######## 1. GET DATA ##########

import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor()])
train_dataset = torchvision.datasets.CIFAR10(root='CIFAR10', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=min_batch_size,
                                          shuffle=True, num_workers=1)
X = torch.tensor(train_dataset.data).type(dtype)
X = torch.flatten(X,1)
y = torch.tensor(train_dataset.targets).type(dtype_labels)

test_dataset = torchvision.datasets.CIFAR10(root='CIFAR10', train=False,
                                        download=True, transform=transform)
X_test = torch.tensor(test_dataset.data).type(dtype)
X_test=X_test.reshape(-1,3*32*32)
y_test = torch.tensor(test_dataset.targets).type(dtype_labels)


######## 2. COMPILE NETS ##########


dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.view(-1,3*32*32)
D_in=images[0].shape[0]
D_out=10 # for mnist is 10 
_layers=np.append(hidden_layers,[D_out])
layers=np.append([D_in], _layers)  #this variable contains the Network arcitecture in terms of units in each layer,e.g. 5,10,10,1 (D_in,hidden 1, hidden 2, D_out)
print('Network architecture (no of units in layer): ',layers)
#networks 

# MLP without batch normalization 
class MlpPlane(torch.nn.Module): 
    def __init__(self,h_sizes): 
        super(MlpPlane,self).__init__()
        self.h_sizes = h_sizes
        self.layers = nn.ModuleList()
        for k in range(len(h_sizes)-1):
            linear_module = nn.Linear(h_sizes[k].item(), h_sizes[k+1].item())
            variance = np.sqrt(2.0/(h_sizes[k].item() + h_sizes[k+1].item()))
            linear_module.weight.data.normal_(0.0, variance)
            self.layers.append(linear_module)
    def forward(self,x):
        for k in range(len(self.h_sizes)-2): 
            x = torch.relu(self.layers[k](x))
        return self.layers[len(self.h_sizes)-2](x)

    def forward_pretrain(self,x):
        for k in range(len(self.h_sizes)-2): 
            x = torch.relu(self.layers[k](x))
        return x

    def get_weights(self): 
        ws = [None]*(len(self.h_sizes)-1)
        for k in range(len(self.h_sizes)-1): 
            ws[k] = self.layers[k].weight
        return ws
    def getlayerloss(self,x,layer_num): 
        for k in range(layer_num+1): 
            x = torch.relu(self.layers[k](x))
        
        M = x.t().mm(x)/x.size(0)
        return x,torch.trace(M.mm(M))/torch.trace(M)**2
    def getblanceloss(self,x):
        lo = 0 
        for k in range(len(self.h_sizes)-1): 
            x = torch.relu(self.layers[k](x))
            M = x.mm(x.t())/float(min_batch_size)
            lo = lo + torch.trace(M.mm(M))/torch.trace(M)**2
        return lo

##### BATCH Normalization  #####
class MlpBatch(MlpPlane): 
    def __init__(self,h_sizes): 
        super(MlpBatch,self).__init__(h_sizes)
        self.batches = nn.ModuleList()
        for k in range(len(h_sizes)-2): 
            self.batches.append(torch.nn.BatchNorm1d(num_features=h_sizes[k+1].item()))#,momentum=0.0
    def forward(self,x):
        for k in range(len(self.h_sizes)-2): 
            x = torch.relu(self.batches[k](self.layers[k](x)))
        return self.layers[len(self.h_sizes)-2](x)   


######## 3. COMPILE TRAINING ROUTINES ##########

import torch.nn.functional as f
def run_training(mlp, epochs = 6,ss = step_size):  
    errors = []
    test_errors = []
    accuracies = []
    criterion = torch.nn.CrossEntropyLoss(size_average=True)
    opt2= torch.optim.SGD(mlp.parameters(),lr =ss )
    loss_epoch = 0 
    data_counter = 0 
    N = X.size(0)
    break_outer=False

    for epoch in range(epochs):  # loop over the dataset multiple times
        rperm = torch.randperm(N).cuda()
        loss_epoch = 0 
        data_counter = 0 
        i = 0 
        accuracy=[]
        while data_counter<N-1:
            opt2.zero_grad()
            fidx = i*(min_batch_size)
            tidx = min((i+1)*(min_batch_size),N-1)
            data_counter = tidx
            inputs = X[rperm[fidx:tidx]]
            labels = y[rperm[fidx:tidx]]
            outputs = mlp.forward(inputs)
            loss = criterion(outputs, labels)
            if math.isnan(loss.item()):
                break_outer=True
            loss.backward()
            opt2.step()
            loss_epoch += loss.item()*inputs.shape[0]/float(N)
            i = i + 1
            accuracy.append(torch.mean(torch.eq(labels, torch.argmax(outputs, 1)).float()).data)
        test_loss=criterion(mlp.forward(X_test), y_test).cpu().item()
        acc=torch.mean(torch.stack(accuracy)).cpu().item()
        accuracies.append(acc)
        print('epoch:',epoch,',loss:',loss_epoch,'accuracy', acc, 'test_loss: ', test_loss)
        errors.append(loss_epoch)
        test_errors.append(test_loss)

        if break_outer:
            break

    return errors,accuracies, test_errors


######## 4. COMPILE FUNCTION FOR REPEATED RUNS ##########


def run_indep(mlp, mlpclass, num_runs = 1,epoch_num = 25):
    # GRID SEARCH STEPSIZE
    stepss = [0.01,0.001]
    our_errors = []
    for si in stepss: 
        print('==============')
        copy_model = mlpclass(layers).cuda()
        copy_model.load_state_dict(mlp.state_dict())
        our_error,our_accuracy, test_errors = run_training(copy_model,epochs = epoch_num,ss = si)
        our_errors.append(our_error)
    best_end = 1000
    best_idx = 0
    for i in range(len(stepss)): 
        if our_errors[i][-1] < best_end: 
            print(our_errors[i][-1],best_end,'#')
            best_end = our_errors[i][-1]
            best_idx = i 
    best_stepsize = stepss[best_idx]

    print('BEST STEPSIZE: ', best_stepsize)    

    # RUN EXPERIMENTS

    errors = []
    accuracies = []
    test_errors = []
    for j in range(num_runs): 
        print('>>>>>>>> new indp. run <<<<<<<<<')
        copy_model = mlpclass(layers).cuda()
        copy_model.load_state_dict(mlp.state_dict())
        error,accuracy,test_error = run_training(copy_model,epochs = epoch_num,ss = best_stepsize)
        errors.append(error)
        accuracies.append(accuracy)
        test_errors.append(test_error)
    return errors,accuracies,test_errors



######## 5. PRE-TRAIN ##########

rperm = torch.randperm(X.size(0)).cuda()

xb = X[rperm[0:min_batch_size]]
bsize = xb.shape[0]
mlp = MlpPlane(layers).cuda()

m = 0 
for k in range(1000):
    fidx = (m+1)*min_batch_size
    tidx = min((m+2)*min_batch_size,X.size(0)-1)
    
    xb = X[rperm[fidx:tidx]]
    mopt = torch.optim.SGD(mlp.parameters(),lr =0.001 )
    for i in range(40): 
        mopt.zero_grad()
        outn = mlp.forward_pretrain(xb)
        M= outn.mm(outn.t())
        loss = torch.norm(M)/torch.trace(M)
        loss.backward()
        mopt.step()
    if k % 100 ==0: 
        print(loss.item())
    m +=1
    if tidx == X.size(0)-1: 
        m = 0
xb = X[rperm[0:32]]
outn = mlp.forward(xb)
M= outn.mm(outn.t())
loss = torch.norm(M)/torch.trace(M)
print('soft rank on random input = ', loss.item())



######## 6. RUN ##########


our_errors, our_accuracies, our_test_loss = run_indep(mlp,MlpPlane,epoch_num = 100)

name = 'result_our_%d_loss' % no_of_hl
np.save(name, our_errors)

name = 'result_our_%d_acc' % no_of_hl
np.save(name, our_accuracies)

name = 'result_our_%d_test_loss' % no_of_hl
np.save(name, our_test_loss)






