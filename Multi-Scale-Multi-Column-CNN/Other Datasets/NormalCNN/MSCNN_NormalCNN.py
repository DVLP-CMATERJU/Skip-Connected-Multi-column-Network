import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import time
import numpy as np
import os.path


def generateNum(start, size):
    Numlist = [start]
    num = start
    for i in range(size-1):
        num = num + 1
        Numlist.append(num)

    return Numlist


num_input_channel = 1
num_class = 10
cv_fold = int(input("Enter folds for cross-validation: "))

resume_weights = "sample_data/checkpoint1.pth.tar"

cuda = torch.cuda.is_available()

torch.manual_seed(1)

if cuda:
    torch.cuda.manual_seed(1)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, ],
    #                      std=[0.5, ])
])


def train_cpu(model, optimizer, train_loader, loss_fun):
    average_time = 0
    total = 0
    correct = 0
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        batch_time = time.time()
        images = Variable(images)
        labels = Variable(labels)

        if cuda:
            images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fun(outputs, labels)

        if cuda:
            loss.cpu()

        loss.backward()
        optimizer.step()

        batch_time = time.time() - batch_time
        average_time += batch_time

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (i + 1) % print_every == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f, Accuracy: %.4f, Batch time: %f'
                  % (epoch + 1,
                     num_epochs,
                     i + 1,
                     len(train_loader),
                     loss.item(),
                     correct / total,
                     average_time / print_every))


def eval_cpu(model, test_loader):
    model.eval()

    total = 0
    correct = 0
    for i, (data, labels) in enumerate(test_loader):
        data, labels = Variable(data), Variable(labels)
        if cuda:
            data, labels = data.cuda(), labels.cuda()

        data = data.squeeze(0)
        labels = labels.squeeze(0)

        outputs = model(data)
        if cuda:
            outputs.cpu()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return correct / total


def save_checkpoint(state, is_best, filename="sample_data/checkpoint.pth.tar"):
    if is_best:
        print("=> Saving a new best")
        torch.save(state, filename)
    else:
        print("=> Validation Accuracy did not improve")


class Skip_Model(nn.Module):
    def __init__(self):
        super(Skip_Model, self).__init__()

        self.layer11 = nn.Sequential(
            nn.Conv2d(num_input_channel, 32, kernel_size=3, stride=2, padding=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.layer12 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer13 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=7, stride=1, padding=(2, 2)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU())

        # concatenated features fc layer

        self.fc0 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.BatchNorm1d(512),
            nn.ReLU())

        # final fc layer

        self.fc_final = nn.Sequential(
            nn.Linear(2816, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU())

        '''self.fc_soft = nn.Sequential(
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU())'''

    def forward(self, x):
        x0 = x.view(-1, self.num_flat_features(x))

        # for first column
        x3 = self.layer11(x)

        x3 = self.layer12(x3)

        x3 = self.layer13(x3)
        x13 = x3.view(-1, self.num_flat_features(x3))

        xz0 = self.fc0(x0)

        # all features concatenation

        xz = torch.cat((xz0, x13), 1)

        # final fc layer

        out = self.fc_final(xz)
        # out = self.fc_soft(out)

        return out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
                

model = Skip_Model()
if cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()
if cuda:
    criterion.cuda()

if os.path.isfile(resume_weights):
    print("=> loading checkpoint '{}' ...".format(resume_weights))
    if cuda:
        checkpoint = torch.load(resume_weights)
    else:
        checkpoint = torch.load(resume_weights, map_location=lambda storage, loc: storage)
    start_epoch = checkpoint['epoch']
    best_accuracy = checkpoint['best_accuracy']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (trained for {} epochs)".format(resume_weights, checkpoint['epoch']))


for fold in range(cv_fold):
    num_epochs = 100
    batch_size = 100
    learning_rate = 0.001
    print_every = 10
    best_accuracy = torch.FloatTensor([0])
    start_epoch = 0

    model = Skip_Model()
    if cuda:
        model.cuda()

    train_set = torchvision.datasets.ImageFolder(root="dataset_name/Train", transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    train_loader2 = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)

    val_set = torchvision.datasets.ImageFolder(root="dataset_name/Val", transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
    val_loader2 = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True)

    test_set = torchvision.datasets.ImageFolder(root="dataset_name/Test", transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    test_loader2 = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

    # print(len(train_loader))
    # print(len(val_loader))
    # print(len(test_loader))

    total_step = len(train_loader)

    for epoch in range(num_epochs):
        print("Fold No: ", fold+1)

        print(learning_rate)

        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

        if learning_rate > 0.00003:
            learning_rate = learning_rate * 0.993

        train_cpu(model, optimizer, train_loader, criterion)
        val_acc = eval_cpu(model, val_loader)
        print('=> Validation set: Accuracy: {:.2f}%'.format(val_acc * 100))

        test_acc = eval_cpu(model, test_loader)
        print('=> Test set: Accuracy: {:.2f}%'.format(test_acc * 100))

        acc = torch.FloatTensor([val_acc])

        is_best = bool(acc.numpy() > best_accuracy.numpy())

        best_accuracy = torch.FloatTensor(max(acc.numpy(), best_accuracy.numpy()))

        save_checkpoint({
            'epoch': start_epoch + epoch + 1,
            'state_dict': model.state_dict(),
            'best_accuracy': best_accuracy
        }, is_best)
