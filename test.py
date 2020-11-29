import os
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.prelu1_2 = nn.PReLU()
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.prelu2_2 = nn.PReLU()
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.prelu3_2 = nn.PReLU()
        self.preluip1 = nn.PReLU()
        self.ip1 = nn.Linear(128*3*3, 128)
        self.ip2 = nn.Linear(128, 1, bias=False)
        self.ip3 = nn.Linear(128, 2, bias=False)

    def forward(self, x, target, is_train=True):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)
        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)
        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 128*3*3)
        feature = self.preluip1(self.ip1(x))
        weight = self.ip2(feature)

        if is_train:
            pos = 10
        else:
            pos = target.shape[0]

        weight = weight.reshape(-1, pos)
        weight = F.softmax(weight, 1)
        weight = weight.view(-1, 1)
        feature = feature * weight
        feature = feature.reshape(-1, pos, 128)
        feature = feature.sum(1)
        target = target.reshape(-1, pos)
        target = (((target == 9) * 1).sum(1) != 0) * 1
        output = F.log_softmax(self.ip3(feature), dim=1)

        if is_train:
            loss = nllloss(output, target)

            return loss
        else:
            return output, weight


def train(model, epoch, optimizer, train_loader):
    model.train()
    print("Training... Epoch = %d" % epoch)
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        loss = model(data, target)

        print("epoch: {}, loss: {}".format(epoch, loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(model, epoch, test_loader):
    model.eval()
    print("Testing... Epoch = %d" % epoch)
    right = 0
    length = 0
    save_index = 0
    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        visible_data = (data*0.3081+0.1307)*255
        visible_data = visible_data.cpu().numpy()

        temp = []
        for i in range(visible_data.shape[0]):
            temp.append(visible_data[i, 0, :, :])
        image = np.concatenate(temp, 1)
        image = image.astype(np.uint8)
        mask = np.zeros(shape=[image.shape[0]+20, image.shape[1]], dtype=np.uint8)
        mask[0:28, :] = image

        pred, weight = model(data, target, False)

        target = target.reshape(-1, len(temp))
        target = (((target == 9) * 1).sum(1) != 0)*1

        max_value, max_index = pred.max(1)
        right += ((max_index == target)*1).sum()
        length += len(pred)

        _, index = weight.max(0)
        if max_index == 1:
            mask[30:46, index*28+2:index*28+26] = 255
        im = Image.fromarray(mask)
        if save_index < 20:
            im.save("out{}.jpeg".format(save_index))
        else:
            break
        save_index += 1
    print(right/length)


def train_and_test():
    # Dataset
    trainset = datasets.MNIST('./data', download=True, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]))
    train_loader = DataLoader(trainset, batch_size=1000, shuffle=True, num_workers=4)

    testset = datasets.MNIST('./data', download=True, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]))
    test_loader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=4)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
    sheduler = lr_scheduler.StepLR(optimizer, 50, gamma=0.1)

    for epoch in range(200):
        sheduler.step(epoch + 1)
        train(model, epoch + 1, optimizer, train_loader)
        if (epoch + 1) % 10 == 0:
            test(model, epoch + 1, test_loader)
            save_path = os.path.join("E:\\Work\\models", "{}.pth".format(epoch + 1))
            torch.save(model.state_dict(), save_path)


def only_test():
    testset = datasets.MNIST('./data', download=True, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]))
    test_loader = DataLoader(testset, batch_size=30, shuffle=True, num_workers=4)

    model = Net().to(device)
    state_dict = torch.load("E:\\Work\\models\\80.pth")
    model.load_state_dict(state_dict)
    test(model, 80, test_loader)


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available() and True
    device = torch.device("cuda" if use_cuda else "cpu")
    only_test()
