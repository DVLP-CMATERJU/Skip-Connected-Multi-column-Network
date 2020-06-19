import torch
import torch.nn as nn
from sklearn import svm
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
import seaborn
import matplotlib.pyplot as plt
import numpy as np
import pickle

num_input_channel = 1
batch_size = 1


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

    
print("Loading the saved loaders...")

train_loader = pickle.load(open("drive/Deep/train_loader2.txt", 'rb'))
val_loader = pickle.load(open("drive/Deep/val_loader2.txt", 'rb'))
test_loader = pickle.load(open("drive/Deep/test_loader2.txt", 'rb'))

print(len(train_loader))
print(len(val_loader))
print(len(test_loader))
print("Dataset is loaded")

resume_weights = "drive/Deep/checkpointBT.pth.tar"
cuda = torch.cuda.is_available()

model = Binary_Tree()
if cuda:
    model.cuda()

checkpoint = torch.load(resume_weights, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['state_dict'])


def flatten(input):
    new_list = []
    for i in input:
        for j in i:
            new_list.append(j)
    return new_list


def eval(model, test_loader):
    output_label = []
    output_target = []
    model.eval()
    for i, (data, target) in enumerate(test_loader):
        data, target = Variable(data, volatile=True), Variable(target)
        if cuda:
            data, target = data.cuda(), target.cuda()

        output = model(data)

        output = output.cpu()
        target = target.cpu()
        output = output.data.numpy()
        target = target.data.numpy()
        output = output.tolist()
        target = target.tolist()
        output = np.ravel(output)
        output_label.append(output)
        output_target.append(target)

    return output_label, output_target


print("Constructing dataset of SVM...")
x_train, y_train = eval(model, train_loader)
pickle.dump(x_train, open('drive/Deep/x_train.txt', 'wb'))
pickle.dump(y_train, open('drive/Deep/y_train.txt', 'wb'))

print("Dataset of SVM is constructed", end='\n\n\n')

print("Loading the dataset for SVM")

save_model = "drive/Deep/svm_model.sav"
x = np.array(pickle.load(open('drive/Deep/x_train.txt', 'rb')))
y = np.array(pickle.load(open('drive/Deep/y_train.txt', 'rb')))

print(np.shape(x))
print(np.shape(y))

print("SVM starts training...")
clf = svm.SVC(kernel='rbf')

y = np.reshape(y, (len(y), 1))
clf.fit(x, y)
training_score = clf.score(x, y)
print("Training Accuracy : ", training_score * 100)
pickle.dump(clf, open(save_model, 'wb'))

clf = pickle.load(open(save_model, 'rb'))
x_val, y_val = eval(model, val_loader)
y_val = np.reshape(y_val, (len(y_val), 1))
val_score = clf.score(x_val, y_val)
print("Validation Accuracy : ", val_score * 100)

print("SVM completed it's training")

print("SVM starts testing...")
clf = pickle.load(open(save_model, 'rb'))
x_test, y_test = eval(model, test_loader)
y_test = np.reshape(y_test, (len(y_test), 1))
test_score = clf.score(x_test, y_test)
print("Testing Accuracy : ", test_score * 100)

y_pred = clf.predict(x_test)
cnf_matrix = confusion_matrix(y_test, y_pred)
pickle.dump(cnf_matrix, open('drive/Deep/confusion_matrix.txt', 'wb'))
np.set_printoptions(precision=2)
print(cnf_matrix)

seaborn.heatmap(cnf_matrix, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

