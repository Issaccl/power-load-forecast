import torch
import torch.utils.data
from torch import optim
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
show = ToPILImage()
import torch.nn as nn
import torch.nn.functional as F


transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


#训练集
trainset=tv.datasets.CIFAR10(
    root='DatasetCNN/',
    train=True,
    download=True,
    transform=transform

)

trainloader=torch.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=2
)


#测试集
testset=tv.datasets.CIFAR10(
    "DatasetCNN/",
    train=False,
    download=True,
    transform=transform
)

testloader=torch.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=False,
    num_workers=2
)

classes=('plane',"car",'bird','cat'
         ,'deer','dog','frog','horse','ship','truck')


dataiter = iter(trainloader)   # trainloader is a DataLoader Object
images, labels = next(dataiter) # 返回4张图片及标签   images,labels都是Tensor    images.size()= torch.Size([4, 3, 32, 32])     lables = tensor([5, 6, 3, 8])
print(' '.join('%11s'%classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid((images+1)/2)).resize((400,100))


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        x=x.view(x.size()[0],-1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)

        return x

net=Net()
print(net)

#损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

#训练部分
torch.set_num_threads(4)
for epoch in range(10):

    running_loss=0.0
    for i,data in enumerate(trainloader,0):

        inputs,labels=data
        optimizer.zero_grad()

        outputs=net(inputs)
        loss=criterion(outputs,labels)
        loss.backward()

        optimizer.step()

        running_loss+=loss.item()
        if i%2000==1999:
            print('[%d %5d] loss:%.3f'\
                  %(epoch+1,i+1,running_loss/2000))

            running_loss=0.0

print('Finished Training')


dataiter = iter(testloader)
images, labels = next(dataiter)
print('实际的label: ', ' '.join(
            '%08s'%classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid((images+1) / 2 )).resize((400,100))

images.shape

outputs=net(images)
_,predicted=torch.max(outputs,1)
print('预测结果：',' '.join('%5s'\
                           %classes[predicted[j]] for j in range(4)))


correct=0
total=0

with torch.no_grad():
    for data in testloader:  # data是个tuple
        images, labels = data  # image和label 都是tensor
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)  # labels tensor([3, 8, 8, 0])            labels.size: torch.Size([4])
        correct += (predicted == labels).sum().item()

print('10000张测试集中的准确率为: %d %%' % (100 * correct / total))































