#Don't change batch size
batch_size = 64

from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(28**2, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

def accuracy(pred, labels):
    correct = (pred.view(-1).long() == labels).sum()
    print(pred.view(-1).long())
    print(labels)
    total = pred.size(0)
    return 100 * (correct.float() / total)

#class LinearSVM(nn.Module):



## USE THIS SNIPPET TO GET BINARY TRAIN/TEST DATA

train_data = datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
test_data = datasets.MNIST('./data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

subset_indices = ((train_data.targets == 0) + (train_data.targets == 1)).nonzero()
train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size, 
  shuffle=False,sampler=SubsetRandomSampler(subset_indices.view(-1)))


subset_indices = ((test_data.targets == 0) + (test_data.targets == 1)).nonzero()
test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size, 
  shuffle=False,sampler=SubsetRandomSampler(subset_indices.view(-1)))

# The number of epochs is at least 10, you can increase it to achieve better performance
num_epochs = 100

loss_func = nn.BCELoss()
model = LogisticRegression()
opt = torch.optim.SGD(model.parameters(), lr=0.01)

# Training the Model
for epoch in range(num_epochs):
    total_loss = 0
    correct, total = 0., 0.
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 28*28)

        #Convert labels from 0,1 to -1,1
        current_batch_size = labels.size(0)
        #labels = (2*(labels.float()-0.5)).long()
        reshaped_labels = labels.float().reshape(current_batch_size, 1)

        pred = model(images)
        loss = loss_func(pred, reshaped_labels)
        loss.backward()
        opt.step()

        total_loss += loss.item()
        temp_corr = (pred.view(-1).long() == labels).sum()
        correct += temp_corr
        total += images.shape[0]
        
    epoch_accuracy = 100 * (correct.float() / total)
    ## Print your results every epoch
    print("Epoch {}: Loss {} Accuracy {}".format(epoch, total_loss, epoch_accuracy))

# Test the Model
correct = 0.
total = 0.
for images, labels in test_loader:
    images = images.view(-1, 28*28)
    
    ## Put your prediction code here, currently use a random prediction
    prediction = model(images)

    correct += (prediction.view(-1).long() == labels).sum()
    total += images.shape[0]
print('Accuracy of the model on the test images: %f %%' % (100 * (correct.float() / total)))


    
