import os
import time
import numpy as np
import needle.nn as nn
import needle as ndl
import sys
sys.path.append('../python')

np.random.seed(0)


class Residual(nn.Module):
    def __init__(self, dim, hidden_dim, norm, drop_prob) -> None:
        super().__init__()
        self.linear0 = nn.Linear(dim, hidden_dim)
        self.norm0 = norm(hidden_dim)
        self.relu0 = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)
        self.linear1 = nn.Linear(hidden_dim, dim)
        self.norm1 = norm(dim)
        self.relu1 = nn.ReLU()

    def forward(self, x: ndl.Tensor):
        new_x = self.linear0(x)
        new_x = self.norm0(new_x)
        new_x = self.relu0(new_x)
        new_x = self.dropout(new_x)
        new_x = self.linear1(new_x)
        new_x = self.norm1(new_x)

        x = new_x+x

        x = self.relu1(x)
        return x


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    # BEGIN YOUR SOLUTION
    return Residual(dim, hidden_dim, norm, drop_prob)
    # END YOUR SOLUTION


class MLPResNetModule(nn.Module):
    def __init__(self, dim, hidden_dim, num_blocks,  num_classes, norm, drop_prob) -> None:
        super().__init__()
        self.linear0 = nn.Linear(dim, hidden_dim)
        self.relu0 = nn.ReLU()
        residuals = []
        dim = hidden_dim
        hidden_dim = hidden_dim//2
        for i in range(num_blocks):
            re = ResidualBlock(dim, hidden_dim, norm, drop_prob)
            residuals.append(re)
        self.residual_seq = nn.Sequential(*residuals)
        self.final_linear = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.linear0(x)
        x = self.relu0(x)
        x = self.residual_seq(x)
        x = self.final_linear(x)
        return x


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    # BEGIN YOUR SOLUTION
    return MLPResNetModule(dim, hidden_dim, num_blocks, num_classes, norm, drop_prob)
    # END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    # BEGIN YOUR SOLUTION
    loss_func = nn.SoftmaxLoss()
    total_loss = 0.
    count = 0
    data_len = 0
    total_error_rate = 0
    if opt is not None:
        model.train()
        for i, data in enumerate(dataloader):
            X, y = data
            opt.reset_grad()
            outputs = model(X.reshape((X.shape[0], -1)))
            preds = np.argmax(outputs.numpy(), axis=1)
            data_len += len(preds)
            total_error_rate += np.sum(preds != y.numpy())
            loss = loss_func(outputs, y)
            total_loss += loss.numpy()
            count += 1
            loss.backward()
            opt.step()
    else:
        model.eval()
        for i, data in enumerate(dataloader):
            X, y = data
            outputs = model(X.reshape((X.shape[0], -1)))
            preds = np.argmax(outputs.numpy(), axis=1)
            data_len += len(preds)
            total_error_rate += np.sum(preds != y.numpy())
            loss = loss_func(outputs, y)
            total_loss += loss.numpy()
            count += 1

        # all_X = []
        # all_y = []
        # for i, data in enumerate(dataloader):
        #     X, y = data
        #     for x in X.numpy():
        #         all_X.append(x)
        #     for y in y.numpy():
        #         all_y.append(y)

        # input_X = ndl.Tensor(np.array(all_X).reshape(len(all_X), -1))
        # outputs = model(input_X)
        # preds = np.argmax(outputs.numpy(), axis=1)
        # avg_error_rate = np.sum(preds != all_y)/len(preds)
        # avg_loss = loss_func(outputs, ndl.Tensor(y))

        # return avg_error_rate, avg_loss.numpy()
    return total_error_rate/data_len, total_loss/count

    # END YOUR SOLUTION


def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    # BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(data_dir+"/train-images-idx3-ubyte.gz",
                                          data_dir+"/train-labels-idx1-ubyte.gz")
    test_dataset = ndl.data.MNISTDataset(data_dir+"/t10k-images-idx3-ubyte.gz",
                                         data_dir+"/t10k-labels-idx1-ubyte.gz")

    train_dataloader = ndl.data.DataLoader(
        train_dataset, batch_size, shuffle=True)
    test_dataloader = ndl.data.DataLoader(
        test_dataset, batch_size, shuffle=True)
    dim = len(train_dataset[0][0].reshape(-1))
    model = MLPResNet(dim, hidden_dim=hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    all_loss = []
    for i in range(epochs):
        train_error_rate, train_loss = epoch(train_dataloader, model, opt)
        test_error_rate, test_loss = epoch(test_dataloader, model)

    return train_error_rate, train_loss, test_error_rate, test_loss


if __name__ == "__main__":
    train_mnist(data_dir="../data")
