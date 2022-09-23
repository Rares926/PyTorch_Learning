from matplotlib import image
import torch 
import torch.nn as nn
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 


#device config
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper parameters
input_size=784 #28x28
hidden_size=100
num_classes=10
num_epochs=2
batch_size=100
learning_rate=0.01

#MNIST
train_dataset=torchvision.datasets.MNIST(root='./assets',
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=True)

test_dataset=torchvision.datasets.MNIST(root='./assets',
                                        train=False,
                                        transform=transforms.ToTensor(),
                                        download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

examples=iter(train_loader)
samples,labels=examples.next()
print(samples.shape,labels.shape)


for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0],cmap='gray')
    plt.title(labels[i].item(),fontdict={'fontsize': 15, 'fontweight': 'medium'})
plt.show()


#define de model 

class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes) -> None:
        super(NeuralNet,self).__init__()
        self.linear1=nn.Linear(input_size,hidden_size)
        self.relu=nn.ReLU()
        self.linear2=nn.Linear(hidden_size,num_classes)
        #we don't have a softmax activation because we will use the crossentropyloss

    def forward(self,x):
        out=self.linear1(x)
        out=self.relu(out)
        out=self.linear2(out)
        return out


model=NeuralNet(input_size,hidden_size,num_classes).to(device)

#loss and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)


#training loop
n_total_steps=len(train_loader)

#loop over epochs
for epoch in range(num_epochs):
    #loop over batches
    for i, (images,labels) in enumerate(train_loader):
    #64 1 28 28 
    #100 784
        images=images.reshape(-1,28*28).to(device) #practic am facut un layer de flatten manual 
        labels=labels.to(device)

        #forward
        outputs=model(images)
        loss=criterion(outputs,labels)

        #backward
        loss.backward()

        #update step
        optimizer.step()

        #clear gradients
        optimizer.zero_grad()

        #print info 
        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')


with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
