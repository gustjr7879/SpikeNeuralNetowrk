#이번에는 어떻게 spike neuron을 순환신경망으로 만들고, 시간에 따라서 역전파, 미분에 대한 이야기를한다(spike neuron) 또한 fully-connected network를 만들어서 MNIST data를 학습시킨다.
#이번 실습에서는 전에 실습한 것들이 다 구성되어있다고 생각하고 이미지 classification을 하도록한다.

import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import itertools
#일반적인 신경망에서는 activation function을 거쳐서 나온 가장 높은 값을 예측으로 사용한다. 하지만 spike를 해석하려면 몇가지 옵션이 있음
# Rate coding : spike수가 가장 높은 뉴런을 예측 클래스로 사용함 / Latency coding : 가장 먼저 발생하는 뉴런을 예측 클래스로 사용함
# 처음에 공부한 내용이 다시나오지만, 차이점은 여기에서 입력데이터를 스파이크로 인코딩, 변환하는 대신에 출력스파이크를 통해서 해석하는 것에 있음(데이터로 쓸건가, 예측값으로 쓸건가)
# rate coding관점에서 본다면 네트워크를 통해서 학습을 하면, 클래스에 해당하는 뉴런이 가장 많은 spike를 발생하는 것이 바람직함
# 이렇게 만들기 위해서 정답 클래스의 membrane potential이 threshold보다 높게만들고, 틀리면 낮게 만들어야함 (여기서 의문 : 그렇다면 -reset에 가중치를 해주면..?)
# loss를 구할때 정답 클래스의 membrane potential은 증가시키고 틀린 class는 감소시킴 이렇게 해서 correct class는 계속 실행되고, 틀리면 실행이 억제됨
# 이 과정은 학습과정(모든 time step)에 적용되기 때문에 모든 step에서 loss가 발생하고 이러한 loss는 학습이 종료될때 함께 합산된다.
# 이를 통해서 spike neural network에 loss function을 적용할 수 있고, snnTorch에서는 snn.functional을 통해서 사용할 수 있다.

batch_size = 128
data_path = './data/mnist'
dtype = torch.float
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

train_loader = DataLoader(mnist_train,batch_size=batch_size,shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test,batch_size=batch_size,shuffle=True,drop_last=True)

# Define The Network
# Network architecture
num_inputs = 28*28
num_hidden = 1000
num_outputs = 10

# Temporal dynamics
num_steps = 25
beta = 0.95

class SpikeNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        #init layer
        self.fc1 = nn.Linear(num_inputs,num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden,num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):

        # init hidden states at t = 0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        #record
        mem_rec = []
        spk_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            #print(cur1[0])
            spk1,mem1 = self.lif1(cur1,mem1)
            #print(spk1[0][1])
            #print(spk1,spk1.shape)
            cur2 = self.fc2(spk1)
            spk2,mem2 = self.lif2(cur2,mem2)
            #print(spk2,spk2.shape)
            #exit()
            spk_rec.append(spk2)
            mem_rec.append(mem2)
        return torch.stack(spk_rec,dim=0), torch.stack(mem_rec,dim=0)
    
model = SpikeNeuralNet().to(device)
#fc1 layer = MNIST dataset을 픽셀별로 넣음
#lif1 = 가중된 데이터를 시간경과에 따라서 통합하고 threshold를 넘으면 spike
#fc2 layer = lif1를 통하여 얻어진 출력 스파이크에 선형변환함
#lif2 = weighted된 spike를 통합함
#아래 코드는 데이터를 통하여 spike를 얻고, 가장 많이 나온 것을 비교함

def print_batch_accuracy(data,targets,train=False):
    output,_ = model(data.view(batch_size,-1))
    _,idx = output.sum(dim=0).max(1)
    acc = np.mean((targets==idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

loss = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr= 5e-4,betas=(0.9,0.999))

data,targets = next(iter(train_loader))
def train_printer():
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")

num_epochs = 1
loss_hist = []
test_loss_hist = []
counter = 0

# Outer training loop
for epoch in range(num_epochs):
    iter_counter = 0
    train_batch = iter(train_loader)

    # Minibatch training loop 미니배치 안에 있는 애들만 가져옴
    
    for data, targets in train_batch:
        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        model.train()
        spk_rec, mem_rec = model(data.view(batch_size, -1)) 

        # initialize the loss & sum over time
        loss_val = torch.zeros((1), dtype=dtype, device=device)
        for step in range(num_steps): #step마다 로스 계산해주기
            loss_val += loss(mem_rec[step], targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

# Test set
total = 0
correct = 0
with torch.no_grad():
    model.eval()
    for data, targets in test_loader:
        test_data = data.to(device)
        test_targets = targets.to(device)

        # Test set forward pass
        test_spk, test_mem = model(test_data.view(test_data.size(0), -1))


        _,predicted = test_spk.sum(dim=0).max(1) #test 전체 합 중 가장 큰것(25time step spike총합)
        # Print train/test loss/accuracy
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

print(f"Total correctly classified test set images: {correct}/{total}")
print(f"Test Set Accuracy: {100 * correct / total:.2f}%")