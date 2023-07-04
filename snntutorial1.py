#spike neural network 공부 및 실습
#snntorch를 이용하여서 쭉 빌드하자

import snntorch as snn
import torch
seed = 42
torch.manual_seed(seed)
#Training parameter
batch_size = 128
data_path = './data/mnist'
num_class = 10 #mnist class

dtype = torch.float

#import dataset -> mnist
from torchvision import datasets,transforms
transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,),(1,))
])
mnist_train = datasets.MNIST(data_path,train=True,download=True,transform=transform)

#large datasets이 필요하지 않다. snntorch.utils에는 데이터셋 수정을 위한 function이 포함되어있다.
# data_subset은 데이터셋을 감소시켜준다.(하위 집합으로 나뉘어진다고 봐도 무방함)
from snntorch import utils
subset = 10
mnist_train = utils.data_subset(mnist_train,subset) # 60,000개의 데이터셋을 10개(6000개씩)으로 나눠서 작게만들어줌
print(f"The Size of Mnist_train is {len(mnist_train)}")

#dataloader로 불러서 batchsize별로 나눠주기
from torch.utils.data import DataLoader
train_loader = DataLoader(mnist_train,batch_size=batch_size,shuffle=True)


# Spike Encoding - SNN은 time-varying data를 사용하게 만들어졌지만 기존 데이터는 time-varying data가 아님
# MNIST 데이터를 SNN에 사용하기 위한 두가지 옵션이 존재함
# 1. 같은 training sample을 반복해서 네트워크에 통과시킴(매 시간마다) 이렇게하면 MNIST데이터를 변하지않는 동영상처럼 만들 수 있고,
# 통과되는 sample data는 0과 1 사이의 정규화된 high precision value가 됨
# 2. input data를 시퀀스 길이(num_step)의 spike train으로 변환해준다. 이렇게 하면 0과 1사이의 불연속 값을 가지게 되고 
# 이렇게 하면 MNIST는 원본 이미지ㅏ와의 관계를 feature로 하는 time-varying 시퀀스로 변환할 수 있다.
# 첫번째 방법은 간단하지만 SNN의 시간적인 부분을 활용하지 않아서, 2의 방법을 자주 사용하고 이를 살펴본다 (data to spike 변환과정)

# snntorch.spikegen 을 사용하면(spike generation) data를 spike로 변환할 수 있다.
# spikegen.rate(rate coding), spikegen.latency(latency coding), spikegen.delta(delta modulation)의 옵션이 존재한다.
# rate coding은 스파이킹 빈도를 결정하고, latency는 스파이크 타이밍을 결정해주고, 델타는 시간적 변화를 사용하여 스파이크를 생성한다.

#첫번째로 rate coding을 알아보자, 이것은 베르누이 시행(동전의 앞면 뒷면처럼 확률이 a , 1-a인 경우)처럼 취급할 수 있고
#시행횟수가 매우 높아지게 된다면 확률은 수렴할 수 밖에 없다.
# MNIST 데이터셋으로 보면 스파이킹이 일어날 확률은 pixel value에 따라서 달라진다. 흰색 픽셀이면 100% , 검정이면 절대 안일어난다.
# input이 0~1사이로 들어오는데, 확률로 쳐서 0이면 없고 1이면 일어나게 한다. 
#실습
from snntorch import spikegen
num_step = 100
data = iter(train_loader) #Iterate minibatch
data_it, target_it = next(data)
#spikeing data
spike_data1 = spikegen.rate(data_it,num_steps=num_step)
#0과 1사이로 떨어져야지 그렇게 안나온다면 확률로 표현할 수 없다. 이런 경우는 확률로 표현할 수 있도록 자동으로 clip(자름)
print(spike_data1.size()) # num_step * batch_size * input_dimension

#visualization
#snntorch는 snntorch.spikeplot을 통하여 시각화 할 수 있다.
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML,display
spike_data_sample = spike_data1[:,0,0]
print(spike_data_sample.size())
#spikeplot.animator는 2-D data로 간단하게 표현해준다. 

#동영상 저장 코드
#fig, ax = plt.subplots()
#anim = splt.animator(spike_data_sample,fig,ax)
#anim.save('spike_mnist_test.mp4')

print(f"The corresponding target is: {target_it[0]}")

#MNIST 데이터셋은 grayscale image로 흰색 텍스트는 모든 step에서 100% 스파이크를 보장한다. 하지만 스파이크 빈도를 조절할 수 있다.
#gain을 활용하여서 스파이크 빈도를 낮추고 다시 해보도록 한다.
spike_data = spikegen.rate(data_it, num_steps=num_step, gain=0.25)

spike_data_sample2 = spike_data[:, 0, 0]
#fig, ax = plt.subplots()
#anim = splt.animator(spike_data_sample2, fig, ax)
print(spike_data_sample.size())
#anim.save('spike_mnist_test.mp4')

#이렇게 하면 확실하게 spike 빈도가 줄어든 것을 확인할 수 있다.
#왜냐하면 gain을 낮게할 경우 완전한 white부분이 줄어들어서 spike의 빈도가 줄어든다. 결론은 확률또한 1/4이 되는 것과 같다.
#다음으로는 시간에 따른 뉴런 활성화값을 시각화할 수 있다. 우선 2-D tensor로 변환해주고 x축을 시간, y축을 활성화 뉴런으로 볼 수 있다.

'''
spike_data_sample2 = spike_data_sample2.reshape((num_step,-1))
#raster plot
fig = plt.figure(facecolor='w',figsize=(10,5))
ax = fig.add_subplot(111)
splt.raster(spike_data_sample2,ax,s=1.5,c='black')
plt.title('Input Layer')
plt.xlabel('Time Step')
plt.ylabel('Neuron Number')
plt.show()
'''

#Rate coding은 몇가지 문제점(의문점)이 존재한다. 피질이 스파이크 rate로 정보를 전체적으로 인코딩한다고 확신할 수 없기 때문임
#Rate coding으로 전체 spike를 설명할 수 없음 해봤자 15%만 설명가능할 뿐 유일한 메커니즘이 아니다.
#Rate coding으로 처리한다면 속도가 매우 떨어지고 비효율적이게 된다.
#하지만 사용된다는 것은 확실하고, 많은 학습의 경우 rate coding을 통하여 spiking시켜서 해결할 수 있게된다. 또 다른 encoding방법들과 함께 사용된다.

#Latency encoding
#temporal code는 뉴런이 spiking하는 시점에 대한 정보를 캡쳐한다. single spike는 단순 빈도에 의존하는 rate code보다 더 많은 의미를 담고있다.
#노이즈에 대한 민감성이 높아지지만, 소비하는 전력을 매우 감소시킬 수 있다.
#spikegen.latency는 이를 할 수 있게 해준다. 전체시간동안 각입력이 최대 한번만 실행할 수 있게함(spike 횟수제한) 1에 가까운 값이면 일찍실행되고, 0이면 쌓인다음에 실행된다.

#함수는 feature의 intensity를 latency code로 변환해준다.
def conver_to_time(data,tau=5,threshold=0.01):
    spike_time = tau*torch.log(data/(data-threshold))
    return spike_time
raw_input = torch.arange(0,5,0.05) # 0부터 5까지 텐서
spike_times = conver_to_time(raw_input)
'''
plt.plot(raw_input,spike_times)
plt.xlabel('input value')
plt.ylabel('spike time(s)')
plt.show()
'''
#이를 통해서 값이 클수록 spike time이 짧고, 값이 낮을수록 spike time이 길다는 것을 볼 수 있었다.(기하급수적인 관계)
# 이 모든 프로세스는 spikegen.latency를 통하여 자동화할 수 있다.
spike_data = spikegen.latency(data_it,num_steps=100,tau=5,threshold=0.01) #변수 tau는 높아질 수록 spike속도가 느려진다. threshold는 spike임계값, 그 전까지는 폐쇄형임. 임계값 이하의 값들은 최종시간 단계에서 할당된다.

'''
fig = plt.figure(facecolor='w',figsize=(10,5))
ax = fig.add_subplot(111)
splt.raster(spike_data[:,0].view(num_step,-1),ax,s=25,c='black')
plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.show()
'''
#이 코드를 통해서 feature중 값이 높은애들이 먼저 spike(fire)되는 것을 확인할 수 있었고 낮은애들은 나중에 되는 것을 볼 수 있다.
#오른쪽에 몰려있는 feature들은 검정색이고 흰색이 먼저 fire되었고 섞인 애매한애들이 중간에 fire되는 것을 볼 수 있다. 즉, 클러스터링해서 확인할 수 있다는 것이다.
#tau값을 올려서 spike time을 늦추거나 linear=True를 활성화해줘서 spike시간을 선형화 할 수 있다.

spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01, linear=True)

'''
fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spike_data[:, 0].view(num_step, -1), ax, s=25, c="black")
plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.show()
'''
#위의 코드를 실행하면 time마다 활성화되는애들이 딱딱 나뉜것을 확인할 수 있다. 하지만 이는 우리가 설정한 time step 100과 다르다. 1초마다 한번씩 발생시켜서 아무것도 안하는 시간만 증가한 것이다.
#이를 해결하기 위해서 tau를 늘리거나 normalize=True를 활성하여서 num_steps의 전체 범위에 걸쳐서 할 수 있게한다.

spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01,
                              normalize=True, linear=True)

'''
fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spike_data[:, 0].view(num_step, -1), ax, s=25, c="black")

plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.show()
'''
#이 결과를 통하여 좀 더 골고루 분포하는 것을 확인할 수 있다. rate coding과 다른것은 희소성이다. spike 횟수 제한을 걸어서 저전력으로 돌릴 수 있게된다.
#이 방법을 통해서 보면 대부분의 spike가(검정색 부분) 마지막에 fire되는 것을 확인할 수 있는데, 이는 MNIST샘플의 어두운 배경에는 유용한 정보가 없다는 말이된다.
#clip=True로 불필요한 정보를 없애버린다.

spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01,
                              clip=True, normalize=True, linear=True)
'''
fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spike_data[:, 0].view(num_step, -1), ax, s=25, c="black")

plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.show()
'''

'''
spike_data_sample = spike_data[:, 0, 0]

fig,ax = plt.subplots()
anim = splt.animator(spike_data_sample,fig,ax)
anim.save('mnist_latency.mp4')
'''
#이렇게 하여서 latency를 통해서 어떻게 spike되는지 확인할 수 있다.

#다음으로는 Delta Modulation을 알아보자. 망막은 적응력이 있다. 처리할 새로운 것이 있을때만 정보를 처리한다는 것이다. 시야의 변화가 없으면 광수용체 세포가 발화하는 경향이 적다.
#즉 이벤트 중심의 뉴런을 만들기 위해서 snntorch.delta를 활용하여서 시계열 텐서를 input으로 할 수 있다.
#threshold가 있을때, input이 positive이지만 크기가 작다면 조금씩 스윙하다가 negative면 spike가 아에 안되게, 일정 값 이상이 되면 spike가 바로 되게 만들수있다.
#즉 입력값에 따라서 값이 작더라도 변화량에 따라서 spike를 유발할 수 있다.
# Convert data
data = torch.Tensor([0, 1, 0, 2, 8, -20, 20, -5, 0, 1, 0])

spike_data = spikegen.delta(data, threshold=4)

# Create fig, ax
'''
fig = plt.figure(facecolor="w", figsize=(8, 1))
ax = fig.add_subplot(111)


# Raster plot of delta converted data
splt.raster(spike_data, ax, c="black")

plt.title("Input Neuron")
plt.xlabel("Time step")
plt.yticks([])
plt.xlim(0, len(data))
plt.show()
'''
#이를 통해서 4,6,8에서 spike가 활성화되는 것을 볼 수 있다. 만약에 negative spike도 활성화해서 확인하고 싶다면 off_spike를 True해주면된다.
print(spike_data) #이렇게 어디에서 spike가 되었는지도 확인할 수 있게된다.

#이렇게 기본적인 snn torch를 다루는 법과 그 의미를 살펴봤고, 이 다음으로는 spike neuron을 어떻게 만들고 사용하는지에 대한 이야기를 한다.
