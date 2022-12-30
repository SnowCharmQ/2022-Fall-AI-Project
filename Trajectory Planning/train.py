import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from sklearn.model_selection import *


class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.hidden2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.out = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.out(x)
        return x


data = torch.load('data.pth')
feature = data['feature']
label = data['label']
X_train, test_data, y_train, test_label = train_test_split(feature, label, test_size=0.1)

input_size = int(feature.shape[1])
output_size = int(torch.unique(label).shape[0])
net = Net(input_size, int(input_size * 1.5), int(input_size * 1.5), output_size)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.5, 0.999))
criterion = nn.CrossEntropyLoss()

epochs = 500
best_accuracy = 0
train_data, val_data, train_label, val_label = train_test_split(X_train, y_train, test_size=0.1)
for epoch in trange(epochs):
    train_output = net(train_data)
    loss = criterion(train_output, train_label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    val_output = net(val_data)
    val_loss = criterion(val_output, val_label)
    val_loss = val_loss.data.item()
    val_pd = torch.max(val_output, 1)[1]
    val_pd = val_pd.data.numpy()
    val_gt = val_label.data.numpy()
    accuracy = float((val_pd == val_gt).astype(int).sum()) / float(val_pd.size)
    print("Train Loss: {}\tVal Loss: {}\tAccuracy: {}".format(loss.data.item(), val_loss, accuracy))
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(net.state_dict(), "model.pkl")
        print("Save the best model in epoch {}".format(epoch))

model = Net(input_size, int(input_size * 1.5), int(input_size * 1.5), output_size)
model.load_state_dict(torch.load("model.pkl"))
test_output = model(test_data)
test_pd = torch.max(test_output, 1)[1]
test_pd = test_pd.data.numpy()
test_gt = test_label.data.numpy()
accuracy = float((test_pd == test_gt).astype(int).sum()) / float(test_pd.size)
print("Correct: {}".format((test_pd == test_gt).astype(int).sum()))
print("Wrong: {}".format((test_pd != test_gt).astype(int).sum()))
print(accuracy)

df = pd.DataFrame({'gt': test_gt, 'pd': test_pd})
df.to_csv("result.csv")
