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

        self.layer21 = nn.Sequential(
            nn.Conv2d(num_input_channel, 32, kernel_size=5, stride=1, padding=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.layer22 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer23 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=(2, 2)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.layer31 = nn.Sequential(
            nn.Conv2d(num_input_channel, 32, kernel_size=7, stride=1, padding=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.layer32 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer33 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=5, stride=1, padding=(2, 2)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU())

        # first column fc layer

        self.fc11 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1024),
            nn.ReLU())

        self.fc12 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.Dropout(0.5),
            nn.BatchNorm1d(2048),
            nn.ReLU())

        self.fc13 = nn.Sequential(
            nn.Linear(2304, 1024),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1024),
            nn.ReLU())

        # second column fc layer

        self.fc21 = nn.Sequential(
            nn.Linear(7200, 2048),
            nn.Dropout(0.5),
            nn.BatchNorm1d(2048),
            nn.ReLU())

        self.fc22 = nn.Sequential(
            nn.Linear(10816, 4096),
            nn.Dropout(0.5),
            nn.BatchNorm1d(4096),
            nn.ReLU())

        self.fc23 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.Dropout(0.5),
            nn.BatchNorm1d(2048),
            nn.ReLU())

        # third column fc layer

        self.fc31 = nn.Sequential(
            nn.Linear(6272, 2048),
            nn.Dropout(0.5),
            nn.BatchNorm1d(2048),
            nn.ReLU())

        self.fc32 = nn.Sequential(
            nn.Linear(16384, 8192),
            nn.Dropout(0.5),
            nn.BatchNorm1d(8192),
            nn.ReLU())

        self.fc33 = nn.Sequential(
            nn.Linear(16384, 8192),
            nn.Dropout(0.5),
            nn.BatchNorm1d(8192),
            nn.ReLU())

        # concatenated features fc layer

        self.fc0 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.BatchNorm1d(512),
            nn.ReLU())

        self.fc1 = nn.Sequential(
            nn.Linear(5120, 2048),
            nn.Dropout(0.5),
            nn.BatchNorm1d(2048),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(14336, 8192),
            nn.Dropout(0.5),
            nn.BatchNorm1d(8192),
            nn.ReLU())

        self.fc3 = nn.Sequential(
            nn.Linear(11264, 8192),
            nn.Dropout(0.5),
            nn.BatchNorm1d(8192),
            nn.ReLU())

        # final fc layer

        self.fc_final = nn.Sequential(
            nn.Linear(18944, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU())

        '''self.fc_soft = nn.Sequential(
            nn.Linear(2048, 50),
            nn.BatchNorm1d(50),
            nn.ReLU())'''

    def forward(self, x):
        x0 = x.view(-1, self.num_flat_features(x))

        # for first column
        x3 = self.layer11(x)
        x11 = x3.view(-1, self.num_flat_features(x3))

        x3 = self.layer12(x3)
        x12 = x3.view(-1, self.num_flat_features(x3))

        x3 = self.layer13(x3)
        x13 = x3.view(-1, self.num_flat_features(x3))

        # for second column
        x5 = self.layer21(x)
        x21 = x5.view(-1, self.num_flat_features(x5))

        x5 = self.layer22(x5)
        x22 = x5.view(-1, self.num_flat_features(x5))

        x5 = self.layer23(x5)
        x23 = x5.view(-1, self.num_flat_features(x5))

        # for third column
        x7 = self.layer31(x)
        x31 = x7.view(-1, self.num_flat_features(x7))

        x7 = self.layer32(x7)
        x32 = x7.view(-1, self.num_flat_features(x7))

        x7 = self.layer33(x7)
        x33 = x7.view(-1, self.num_flat_features(x7))

        # features from first column

        x11z = self.fc11(x11)
        x12z = self.fc12(x12)
        x13z = self.fc13(x13)

        # features from second column

        x21z = self.fc21(x21)
        x22z = self.fc22(x22)
        x23z = self.fc23(x23)

        # features from third column

        x31z = self.fc31(x31)
        x32z = self.fc32(x32)
        x33z = self.fc33(x33)

        # all concatenated features

        x1 = torch.cat((x11z, x21z), 1)
        x1 = torch.cat((x1, x31z), 1)

        x2 = torch.cat((x12z, x22z), 1)
        x2 = torch.cat((x2, x32z), 1)

        x3 = torch.cat((x13z, x23z), 1)
        x3 = torch.cat((x3, x33z), 1)

        # concatenated features fc layer

        xz0 = self.fc0(x0)
        xz1 = self.fc1(x1)
        xz2 = self.fc2(x2)
        xz3 = self.fc3(x3)

        # all features concatenation

        xz = torch.cat((xz0, xz1), 1)
        xz = torch.cat((xz, xz2), 1)
        xz = torch.cat((xz, xz3), 1)

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
