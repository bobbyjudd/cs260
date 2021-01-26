#Don't change batch size
batch_size = 64

import math
import sys
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

model_types = set(["lr", "svm"])
opt_types = set(["sgd", "sdg-mom"])

def print_usage():
    print("USAGE: python hw1.py <model_type> <optimizer>")
    print("\t<model_type>:  lr, svm")
    print("\t<optimizer> : sgd, sdg-mom")

def convert_plus_minus_one(t):
    return (2*(t.float()-0.5))

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(28**2, 1)

    def forward(self, x):
        y_pred = convert_plus_minus_one(torch.sigmoid(self.linear(x)))
        return y_pred

class LinearSVM(nn.Module):
    def __init__(self):
        super(LinearSVM, self).__init__()
        self.linear = torch.nn.Linear(28**2, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

def correct_count(pred, labels):
    return (pred.view(-1).long() == labels).sum()

def logreg_correct(pred, labels):
    return correct_count(pred, labels)

def svm_correct(pred, labels):
    # Clamp values < 0 to -1 and values > 0 to 1
    pred[pred < 0] = -1
    pred[pred > 0] = 1
    return correct_count(pred, labels)

def svm_loss(output, target):
    loss = torch.mean(torch.clamp(1 - output * target, min=0))  # hinge loss
    return loss

def validation_accuracy(model, test_loader):
    # Test the Model
    correct = 0.
    total = 0.
    for images, labels in test_loader:
        images = images.view(-1, 28*28)
        labels = (2*(labels.float()-0.5)).long()
        ## Put your prediction code here, currently use a random prediction
        prediction = model(images)

        
        if type(model) == LogisticRegression:
            correct += logreg_correct(prediction, labels)
        else:
            correct += svm_correct(prediction, labels)
        total += images.shape[0]
    return 100 * (correct.float() / total)

def run_model(model_type, opt_type):
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
    num_epochs = 10

    # Latex coordinates for plotting loss and accuracy
    loss_coords = ""
    train_coords = ""
    test_coords = ""

    if model_type == "svm":
        model = LinearSVM()
    else:
        model = LogisticRegression()

    if opt_type == "sdg":
        opt = torch.optim.SGD(model.parameters(), lr=1e-6)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9)

    mse_loss = nn.MSELoss()

    # Training the Model
    for epoch in range(num_epochs):
        total_loss = 0
        correct, total = 0., 0.
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 28*28)

            #Convert labels from 0,1 to -1,1
            current_batch_size = labels.size(0)
            labels = convert_plus_minus_one(labels)
            reshaped_labels = labels.float().reshape(current_batch_size, 1)
            
            pred = model(images)

            if type(model) == LogisticRegression:
                loss = mse_loss(pred, reshaped_labels)
            else:
                loss = svm_loss(pred, reshaped_labels)
            
            loss.backward()  # Backpropagation
            opt.step()  # Optimize and adjust weights
            
            if type(model) == LogisticRegression:
                temp_corr = logreg_correct(pred, labels)
            else:
                temp_corr = svm_correct(pred, labels)

            total_loss += loss.item()
            correct += temp_corr
            total += images.shape[0]
            
        train_accuracy = 100 * (correct / total)
        valid_accuracy = validation_accuracy(model, test_loader)
        loss_coords += "({},{:.2f})".format(epoch, total_loss)
        train_coords += "({},{:.2f})".format(epoch, train_accuracy)
        test_coords += "({},{:.2f})".format(epoch, valid_accuracy)
        ## Print your results every epoch
        print("Epoch {}: Loss {} Accuracy {}".format(epoch, total_loss, train_accuracy))
        print('Validation accuracy of the model on the test images: %f %%' % valid_accuracy)

    print("Loss: ",loss_coords)
    print("Train: ",train_coords)
    print("Test: ",test_coords)
    print('Final validation accuracy: %f %%' % validation_accuracy(model, test_loader))

if __name__== "__main__":
    if(len(sys.argv) < 3):
        print_usage()
    else:
        model_type = sys.argv[1]
        opt_type = sys.argv[2]

        if model_type not in model_types:
            print(inval)
        elif opt_type not in opt_types:
            print()
        else:
            run_model(model_type, opt_type)

    
