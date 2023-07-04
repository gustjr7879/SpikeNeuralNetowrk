#이번에는 leaky integrate-and-fire(LIF) 뉴런 모델을 배우고, snntorch에서 이를 어떻게 구성을 하는지 본다.
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

#다양한 뉴런 모델들이 존재한다. hodgkin-huxley 뉴런모델은 생물학적으로 높은 정확도로 전기적인 결과를 재현할 수 있지만 복잡하여서 사용하기 어려움
#ANN 많이 사용되는 방법 weight matrix와 activation function을 활용함
#LIF 뉴런모델 -> ANN과 마찬가지로 가중 입력의 합을 한다. 하지만 직접 activation function에 전달하는 대신 RC회로와 같이 시간을 고려하여서 통합한다.
#통합된 값이 임계값을 넘어서면 fire된다. 정보는 spike내에 저장되는 것이 아니라 spike의 타이밍이나 빈도에 따라서 저장된다. 다양한 버전의 LIF 모델이 있고 snntorch에는
#lapicque's RC model, 1st-order model, synaptic conductance-based neuron model, recurrent 1st-order model, recurrent synaptic conductance-based neuron model, alpha neuron model
#등이 존재한다.
#우리 뇌에서 뉴런은 1000개에서 1만여개의 다른 뉴런들과 연결이 되어있고 spike한다면 downhill뉴런들은 다 느낄 수 있다. 어떤 뉴런이 첫번째로 spike되는지 정할 수 있을까?
#이전 실험(뇌과학 실험)에서 뉴런이 입력에서 충분한 자극을 얻으면 자체적으로 spike를 발사한다는 것이 밝혀짐
#자극은 다른 뉴런, 그리고 인위적으로 자극되는 침습성전극에서 주로 온다. 뉴런은 뉴런끼리 시냅스를 통해서 연결이 되어있는데, 강하게 연결된 경우 자극도 강하게 전달된다.
#또한 자극은 일제히 같은 뉴런에 도착하지 않는다. 시간 역학적으로 분석해야한다.

#lapicque's LIF Neuron model
time_step = 1e-3
num_steps = 100
R = 5
C = 1e-3
#leaky integrate and fire neuron, tau = 5e-3
lif1 = snn.Lapicque(R=R,C=C,time_step=time_step)
#input = spk_in, mem 으로 구성되어있음. 
mem = torch.ones(1)*0.9 #0.9V 이고 potential임
cur_in = torch.zeros(num_steps) #I=0 for all
spk_out = torch.zeros(1) #initialize output spikes
mem_rec = [mem] #store mem potential
for step in range(num_steps):
    spk_out,mem = lif1(cur_in[step],mem)
    mem_rec.append(mem)

mem_rec = torch.stack(mem_rec)
#print(mem_rec)

#이런저런 방법으로 리셋시킨다.
def leaky_integrate_and_fire(mem,x,w,beta,threshold=1):
    spk = (mem>threshold) #만약에 threshold를 초과하면 1, 아니면 0 반환
    mem = beta*mem + w*x - spk*threshold #reset시켜주는 코드가 spk*threshold임
    return spk,mem

delta_t = torch.tensor(1e-3)
tau = torch.tensor(5e-3)
beta = torch.exp(-delta_t/tau)
print(f'decay rate is : {beta:.3f}')

num_steps = 200
#initialize inputs/outputs + small step current input
x = torch.cat((torch.zeros(10),torch.ones(190)*0.5),0) # 10개의 0값, 190개의 0.5값이 input으로 들어감
mem = torch.zeros(1)
spk_out = torch.zeros(1) #mem의 텐서와 spk out의 텐서를 지정해줌
mem_rec = []
spk_rec = []
#neuron parameter
w = 0.4
beta = 0.819

for step in range(num_steps):
    spk,mem = leaky_integrate_and_fire(mem,x[step],w=w,beta=beta)
    mem_rec.append(mem)
    spk_rec.append(spk)

mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)
#이렇게 해서 시각화하게되면 11번째 인덱스부터 값이 들어가게되고 1(threshold)를 달성하면 다시 낮춘다.(reset)그리고 spk값은 T/F로 나온다
#print(spk_rec)
#print(mem_rec)

#이 모델을 snntorch에서 바로 사용가능하다
lif1 = snn.Leaky(beta=0.8)
#input으로는 cur_in (w*x[t])의 개별 요소가 들어간다, mem 이전 상태의 멤브레인 포텐셜이 들어간다.
#output으로는 spk_out과 mem이 나오게된다

#모든 데이터는 torch.tensor로 들어가야하고 이 코드에서는 이미 weighted되어있다고 가정한다.
w = 0.21 #weighted
cur_in = torch.cat((torch.zeros(10),torch.ones(190)*w),0)
mem = torch.zeros(1)
spk = torch.zeros(1)
mem_rec = []
spk_rec = []

for step in range(num_steps):
    spk,mem = lif1(cur_in[step],mem)
    mem_rec.append(mem)
    spk_rec.append(spk)

mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)
#print(spk_rec)

#방금까지 단일 뉴런에 대한 snn을 공부해봤다. 이제는 feedforward spiking neural network를 공부해볼 차례이다.
#snntorch에서는 이를 가능하게 해준다. 이번에는 3-layer snn을 구성하는 실습을 해볼 것이고 784 -> 1000 -> 10을 해본다.

num_inputs = 784
num_hidden = 1000
num_output = 10
beta = 0.99

fc1 = nn.Linear(num_inputs,num_hidden)
lif1 = snn.Leaky(beta=beta)
fc2 = nn.Linear(num_hidden,num_output)
lif2 = snn.Leaky(beta=beta)

#이렇게 신경망들을 선언해주고, hidden 변수와 뉴런들의 spike output을 initial해야한다. 가장 간단한 방법은 .init_leaky()를 사용하는것이다. snntorch의 모든 뉴런들은 init __ 를 사용하여서 시작할 수 있다.
mem1 = lif1.init_leaky()
mem2 = lif2.init_leaky()

mem2_rec = []
spk1_rec = []
spk2_rec = []
#200 time step을 할거고, 784차원이지만, batch_size가 되므로 snntorch에서는 time*batchsize*feature dimension이 된다.
#unsqueeze dim=1을 사용하여서 one batch data만 사용할 수 있다. 200*1*784
spk_in = spikegen.rate_conv(torch.rand((200,784))).unsqueeze(1)
#pytorch와 snntorch가 함께 작동하는 방식은 pytorch가 뉴런들을 함께 실행하고 snntorch가 그 결과를 스파이킹 뉴런 모델에 업로드해서 output을 뽑아내는것이다.
#이는 활성화함수처럼 취급될 수 있다.
# network simulation
for step in range(num_steps):
    cur1 = fc1(spk_in[step]) # post-synaptic current <-- spk_in x weight
    spk1, mem1 = lif1(cur1, mem1) # mem[t+1] <--post-syn current + decayed membrane
    cur2 = fc2(spk1)
    spk2, mem2 = lif2(cur2, mem2)

    mem2_rec.append(mem2)
    spk1_rec.append(spk1)
    spk2_rec.append(spk2)

# convert lists to tensors
mem2_rec = torch.stack(mem2_rec)
spk1_rec = torch.stack(spk1_rec)
spk2_rec = torch.stack(spk2_rec)
#print(spk1_rec[0])
#print(spk2_rec[-5])
#이렇게 해서 나온 spike는 의미를 가지고 있지 않는다.(죄다 랜덤으로 initial했기때문에)
#spikeplot.spike_count는 output레이어에서 spike의 횟수를 count해준다.