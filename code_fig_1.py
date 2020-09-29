import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as func
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import IPython
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.keras.datasets import mnist, fashion_mnist

import pandas as pd


GPU=True

if GPU:
    dtype = torch.cuda.FloatTensor
    dtype_long=torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_long=torch.LongTensor

   ### LOAD DATA
(x_train, y_train), (x_test, y_test) =fashion_mnist.load_data() #fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


x_train = x_train.reshape([-1, 28 ** 2])
x_test = x_test.reshape([-1, 28 ** 2])
full_samples = x_train.shape[0] 
full_samples_test= x_test.shape[0]
X, Y = torch.from_numpy(x_train[:full_samples]).type(dtype), torch.from_numpy(y_train[:full_samples]).type(dtype_long)
X_test, Y_test = torch.from_numpy(x_test[:full_samples_test]).type(dtype), torch.from_numpy(y_test[:full_samples_test]).type(dtype_long)



loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

def run(mymodel,lr):
    losses, grad_collector,accuracy_train,accuracy_train_batch,output_angles,grad_angles,ranks,output_ranks=[],[],[],[],[],[],[],[]
    optimizer = torch.optim.SGD(mymodel.parameters(), lr=lr)
    print("lr: " + str(lr))

    for i in range(iterations):
        optimizer.zero_grad()

        # 2.1 compute loss
        print("\r{} {}.".format(i, "  "), end=" "*10, flush=True)   
        batch=torch.LongTensor(batchsize).random_(0, x_train.shape[0])
        mymodel.train()
        last_layer_activation,y_pred=mymodel.forward(X[batch])
        loss=loss_fn(y_pred,Y[batch])
        loss.backward(retain_graph=True)  #does result change with retaining graph?
        if loss.item()==float('Inf'):
            print('inf')
            raise ValueError('loss is inf')
        if i%stats_every==0:
            losses.append(loss.item())

        if i%stats_every==0: 
	         	output_angles.append(0) #old
        # 2.3 compute grads and update W
        if i==0:
            last_layer_cov=torch.mm(last_layer_activation.T,last_layer_activation)
            singular_values=torch.svd(last_layer_activation)[1]

            stable_rank=torch.norm(last_layer_cov, p='fro')**2/singular_values[0]**4   # STABLE RANK
            h_rank=torch.trace(last_layer_cov)**2/torch.norm(last_layer_cov, p='fro')**2

            output_ranks.append(h_rank.detach().cpu().numpy())
            current_rank=torch.matrix_rank(last_layer_activation/torch.norm(last_layer_activation))
            ranks.append(current_rank)
            
            print("rank"+str(current_rank.item())) 
        
        optimizer.step()
        
        # 2.2 save accuracies
        if i%stats_every==0:
            
            accuracy_train_batch.append(torch.mean(torch.eq(Y[batch], torch.argmax(y_pred, 1)).type(dtype)).data)
            eval_batch=torch.LongTensor(1000).random_(0, x_train.shape[0])
            mymodel.eval()
            with torch.no_grad():
                _stuff, y_pred_full=mymodel.forward(X[eval_batch])
            current_acc=torch.mean(torch.eq(Y[eval_batch], torch.argmax(y_pred_full, 1)).type(dtype)).data
            accuracy_train.append(current_acc)
            
            if i>0:
                print("final_accuracy"+str(current_acc.item()))


    return losses,accuracy_train,ranks



def run_bn(mymodel,lr):
    losses, grad_collector,accuracy_train,accuracy_train_batch,output_angles,grad_angles,ranks,output_ranks=[],[],[],[],[],[],[],[]
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=lr)
    print("lr: " + str(lr))

    for i in range(iterations):
        optimizer.zero_grad()

        # 2.1 compute loss
        print("\r{} {}.".format(i, "  "), end=" "*10, flush=True)   
        batch=torch.LongTensor(batchsize).random_(0, x_train.shape[0])
        mymodel.train()
        means_list,stds_list,last_layer_activation,y_pred=mymodel.forward(X[batch])
        loss=loss_fn(y_pred,Y[batch])
        loss.backward(retain_graph=True)  #does result change with retaining graph?
        
        if loss.item()==float('Inf'):
            print('inf')
            raise ValueError('loss is inf')
        if i%stats_every==0:
            losses.append(loss.item())
            # 2.2 save accuracies
            current_batch_acc=torch.mean(torch.eq(Y[batch], torch.argmax(y_pred, 1)).type(dtype)).data
            accuracy_train_batch.append(current_batch_acc)
            eval_batch=torch.LongTensor(1000).random_(0, x_train.shape[0])
            mymodel.eval()
            with torch.no_grad():
                _stuff1,_stuff2,_stuff3, y_pred_full=mymodel.forward(X[eval_batch])
            current_acc=torch.mean(torch.eq(Y[eval_batch], torch.argmax(y_pred_full, 1)).type(dtype)).data
            accuracy_train.append(current_acc)
            
            if i>0:
                print("final_accuracy"+str(current_acc.item()))
        # 2.2.1 save output angles
     
        output_angles.append(0)
        if i%stats_every==0:
            print("current acc batch: "+str(torch.mean(torch.eq(Y[batch], torch.argmax(y_pred, 1)).type(dtype)).data))
            last_layer_cov=torch.mm(last_layer_activation.T,last_layer_activation)
            singular_values=torch.svd(last_layer_activation)[1]
            stable_rank=torch.norm(last_layer_cov, p='fro')**2/singular_values[0]**4   # STABLE RANK
            h_rank=torch.trace(last_layer_cov)**2/torch.norm(last_layer_cov, p='fro')**2
            
            #ranks.append(stable_rank.detach().numpy())
            output_ranks.append(h_rank.detach().cpu().numpy())
            current_rank=torch.matrix_rank(last_layer_activation/torch.norm(last_layer_activation))
            ranks.append(current_rank)
            print("rank"+str(current_rank.item())) 

            grad_angles.append(0)
        optimizer.step()
     
    
    return losses,accuracy_train,ranks

import torch.nn as nn
import torch
def Flinear(x): # activation funtion
    return torch.relu(x)

def normalize_center(A):
    d=A.shape[1]
    means=torch.mean(A,dim=0)
    stds=torch.std(A,dim=0)
    stds=stds#+0.001
    A_scaled=(A-means.reshape(1,d))/stds.reshape(1,d) ## normalized along COLS (feature lives in row for me)
    return means.detach(), stds.detach(), A_scaled

def normalize_center_mid(A,alpha):
    d=A.shape[1]
    means=torch.mean(A,dim=0)
    means=alpha*means+(1-alpha)*torch.zeros(d)
    stds=torch.std(A,dim=0)
    stds=stds#+0.001
    stds=alpha*stds+(1-alpha)*torch.ones(d)
    A_scaled=(A-means.reshape(1,d))/stds.reshape(1,d) ## normalized along COLS (feature lives in row for me)
    return means.detach(), stds.detach(), A_scaled

def normalize_center_eval(A,means,stds):
    d=A.shape[1]
    A_scaled=(A-means.reshape(1,d))/stds.reshape(1,d) ## normalized along COLS (feature lives in row for me)
    return A_scaled
    
def normalize(A):
    d=A.shape[1]

    stds=torch.std(A,dim=0)
    A_scaled=(A)/stds.reshape(1,d) ## normalized along COLS (feature lives in row for me)
    return 0,0,A_scaled

def center(A):
    d=A.shape[1]
    means=torch.mean(A,dim=0)
    A_scaled=A-means.reshape(1,d)
    return A_scaled


class FFN(nn.Module): #note that the actual number of hidden laers is no_of_hidden_layers+1
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10,no_of_hidden_layers=0, seed=1, act=Flinear):
        super(FFN, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.Win = nn.Linear(input_dim, hidden_dim,bias=True)
        self.layers = torch.nn.ModuleList([nn.Linear(hidden_dim, hidden_dim,bias=True) for _ in range(no_of_hidden_layers)])
        self.Wout = nn.Linear(hidden_dim, output_dim,bias=True)
        self.act=act
        
    def forward(self, input):
        _x = self.Win(input)
        x=self.act(_x)
        for layer in self.layers:
            _x=layer(x)
            x=self.act(_x)
        y_pred = self.Wout(x)
        return x,y_pred
    
    
class BNN(nn.Module): #note that the actual number of hidden laers is no_of_hidden_layers+1
    def __init__(self,broken=False, input_dim=784, hidden_dim=128, output_dim=10,no_of_hidden_layers=0, seed=1, act=Flinear):
        super(BNN, self).__init__()
        self.broken=broken
        if seed is not None:
            torch.manual_seed(seed)
        self.Win = nn.Linear(input_dim, hidden_dim,bias=False)
        self.layers = torch.nn.ModuleList([nn.Linear(hidden_dim, hidden_dim,bias=False) for _ in range(no_of_hidden_layers)])
        self.Wout = nn.Linear(hidden_dim, output_dim,bias=False)
        self.act=act
        if broken:
            print("breaking BatchNorm")
            for w in self.parameters():
                w.data=2*torch.abs(w.data)

    def forward(self, input):
        means_list=[]
        stds_list=[]
        if self.broken:
            means,stds,_x = normalize_center(self.Win(input))
        else:
            means,stds,_x = normalize_center(self.Win(input))
        x=self.act(_x)
    
        for layer in self.layers:
            if self.broken:
                means,stds,_x=normalize_center(layer(x))
            else:
                means,stds,_x=normalize_center(layer(x))

            means_list.append(means)
            stds_list.append(stds)
            x=self.act(_x)
        
        y_pred = self.Wout(x)
        return means_list,stds_list,x,y_pred
    
    def forward_eval(self, input,means_list,stds_list):
        _x = self.Win(input)
        x=self.act(_x)
        k=0
        for layer in self.layers:
            _x=normalize_center_eval(layer(x),means_list[k],stds_list[k])
            x=self.act(_x)
            k=k+1
        
        y_pred = self.Wout(x)
        return x,y_pred
    
def run_and_gun(model_type):  
    loss,accuracy,ranks=[],[],[]
    rank_after_500=[]
    rank_final=[]
    lr=0.01
    for hl in np.arange(0,max_depth):
        if hl%2==0:
            if model_type=="FFN":
                mymodel=FFN(seed=myseed,no_of_hidden_layers=hl).cuda()
                myloss,myaccuracy,myranks=run(mymodel,lr)

            elif model_type=="BNN":
                if True: #these stepsizes have been grid-searched
                    if hl==8:
                        lr=0.05
                    if hl==22:
                        lr=0.005
                    if hl==30:
                        lr=0.0005
                mymodel=BNN(seed=myseed,no_of_hidden_layers=hl,broken=False).cuda()
                myloss,myaccuracy,myranks=run_bn(mymodel,lr)
                
            elif model_type=="broken_BNN":
                mymodel=BNN(seed=myseed,no_of_hidden_layers=hl,broken=True).cuda()
                myloss,myaccuracy,myranks=run_bn(mymodel,lr)
            else:
                print("MODEL UNKNOWN. NOT RUNNING")
                
            loss.append(myloss[-1]), accuracy.append(myaccuracy[-1].item()),ranks.append(myranks[0].item())
            rank_final.append(myranks[-1].item())
            rank_after_500.append(myranks[1].item())
            print(hl)
    print("done")
    loss=pd.DataFrame(data=loss, columns=['value'])
    accuracy=pd.DataFrame(data=accuracy, columns=['value'])
    ranks=pd.DataFrame(data=ranks, columns=['value'])
    rank_after_500=pd.DataFrame(data=rank_after_500, columns=['value'])
    rank_final=pd.DataFrame(data=rank_final, columns=['value'])

    if model_type=="broken_BNN":
        return loss,accuracy,ranks,rank_after_500,rank_final
    else:
        return loss,accuracy,ranks

def run_one():
    hl=32
    lr=0.1
    myseed=torch.LongTensor(1).random_(0, 100) 

    #mymodel=FFN(seed=myseed,no_of_hidden_layers=hl).cuda()
    mymodel=BNN(seed=myseed,no_of_hidden_layers=hl,broken=True).cuda()

    myloss,myaccuracy,myranks=run_bn(mymodel,lr)

def Flinear(x): #  activation funtion
    return torch.relu(x)

batchsize=128
iterations_per_epoch=int(60000/128) #468
epochs=75
iterations=epochs*iterations_per_epoch #3501

max_depth=33
stats_every=iterations-1  #we only care about start and end now

if True:         ## set this TRUE if you only want to try ONE specific network. Specfiy details in def run_one().
                 ## If False, many runs are done to creae the plots (takes very long)
    run_one()    
else:
    #####################
    ######## FFN ########
    #####################
    myseed=torch.LongTensor(1).random_(0, 100) #83 seems to be the magic seed
    loss_1, accuracy_1,ranks_1=run_and_gun('FFN')
    accuracy_1["run_id"],loss_1["run_id"],ranks_1["run_id"]=1,1,1

    myseed=torch.LongTensor(1).random_(0, 100) #83 seems to be the magic seed
    loss_2, accuracy_2,ranks_2=run_and_gun('FFN')
    accuracy_2["run_id"],loss_2["run_id"],ranks_2["run_id"]=2,2,2

    myseed=torch.LongTensor(1).random_(0, 100) #83 seems to be the magic seed
    loss_3, accuracy_3,ranks_3=run_and_gun('FFN')
    accuracy_3["run_id"],loss_3["run_id"],ranks_3["run_id"]=3,3,3

    acc_FFN=pd.concat([accuracy_1, accuracy_2, accuracy_3], sort=False) 
    loss_FFN=pd.concat([loss_1,loss_2,loss_3], sort=False) 
    ranks_FFN=pd.concat([ranks_1,ranks_2,ranks_3], sort=False) 

    acc_FFN.to_pickle("acc_FFN.pkl")
    ranks_FFN.to_pickle("ranks_FFN.pkl")

    #####################
    ######## BNN ########
    #####################    
    myseed=torch.LongTensor(1).random_(0, 100) #83 seems to be the magic seed
    loss_1, accuracy_1,ranks_1=run_and_gun('BNN')
    accuracy_1["run_id"],loss_1["run_id"],ranks_1["run_id"]=1,1,1

    myseed=torch.LongTensor(1).random_(0, 100) #83 seems to be the magic seed
    loss_2, accuracy_2,ranks_2=run_and_gun('BNN')
    accuracy_2["run_id"],loss_2["run_id"],ranks_2["run_id"]=2,2,2

    myseed=torch.LongTensor(1).random_(0, 100) #83 seems to be the magic seed
    loss_3, accuracy_3,ranks_3=run_and_gun('BNN')
    accuracy_3["run_id"],loss_3["run_id"],ranks_3["run_id"]=3,3,3

    acc_BNN=pd.concat([accuracy_1, accuracy_2, accuracy_3], sort=False) 
    loss_BNN=pd.concat([loss_1,loss_2,loss_3], sort=False) 
    ranks_BNN=pd.concat([ranks_1,ranks_2,ranks_3], sort=False) 

    acc_BNN.to_pickle("acc_BNN_3.pkl")
    ranks_BNN.to_pickle("ranks_BNN_3.pkl")

    ############################
    ######## Broken BNN ########
    ############################

    myseed=torch.LongTensor(1).random_(0, 100) #83 seems to be the magic seed
    loss_1, accuracy_1,ranks_1,rank_after_500_1,rank_final_1=run_and_gun('broken_BNN')
    accuracy_1["run_id"],loss_1["run_id"],ranks_1["run_id"],rank_after_500_1["run_id"],rank_final_1["run_id"]=1,1,1,1,1

    myseed=torch.LongTensor(1).random_(0, 100) #83 seems to be the magic seed
    loss_2, accuracy_2,ranks_2,rank_after_500_2,rank_final_2=run_and_gun('broken_BNN')
    accuracy_2["run_id"],loss_2["run_id"],ranks_2["run_id"],rank_after_500_2["run_id"],rank_final_2["run_id"]=2,2,2,2,2

    myseed=torch.LongTensor(1).random_(0, 100) #83 seems to be the magic seed
    loss_3, accuracy_3,ranks_3,rank_after_500_3,rank_final_3=run_and_gun('broken_BNN')
    accuracy_3["run_id"],loss_3["run_id"],ranks_3["run_id"],rank_after_500_3["run_id"],rank_final_3["run_id"]=3,3,3,3,3

    acc_broken_BNN=pd.concat([accuracy_1, accuracy_2, accuracy_3], sort=False) 
    loss_broken_BNN=pd.concat([loss_1,loss_2,loss_3], sort=False) 
    ranks_broken_BNN=pd.concat([ranks_1,ranks_2,ranks_3], sort=False) 
    ranks_after500_broken_BNN=pd.concat([rank_after_500_1,rank_after_500_2,rank_after_500_3], sort=False) 
    final_ranks_broken_BNN=pd.concat([rank_final_1,rank_final_2,rank_final_3], sort=False) 
    acc_broken_BNN.to_pickle("acc_broken_BNN_001.pkl")
    ranks_broken_BNN.to_pickle("ranks_broken_BNN_001.pkl")
    final_ranks_broken_BNN.to_pickle("final_ranks_broken_BNN_001.pkl")
    ranks_after500_broken_BNN.to_pickle("ranks_after_500_broken_BNN_001.pkl")

