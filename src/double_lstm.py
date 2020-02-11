from generate_train_data import generate_train_test_local
import numpy as np
np.set_printoptions(threshold=np.inf) 
import torch
from torch import nn
from torch.utils import data
import torch.nn.functional as F

EPOCH = 20              # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 52         # rnn input size / image width
LR = 0.01               # learning rate
    
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=256,         # rnn hidden unit
            num_layers=2,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            dropout=0.1,
        )
        self.dense1 = nn.Linear(in_features=256,out_features=128)
        self.dense2 = nn.Linear(128,64)
        self.dense3 = nn.Linear(64,34)
        self.act = nn.Sigmoid()

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n,c_n) = self.rnn(x)
        r_out = self.dense1(F.relu(F.dropout(r_out.data,p=0.1)))
        r_out = self.dense2(F.relu(F.dropout(r_out.data,p=0.1)))
        r_out = self.act(self.dense3(F.dropout(r_out.data,p=0.1)))
        out = r_out[:,-1,:]
        return out

if __name__ == "__main__":
    x_train, y_train = generate_train_test_local()
    TIME_STEP = x_train[0].shape[0]
    
    tensor_x = torch.Tensor(x_train)
    tensor_y = torch.Tensor(y_train)

    my_dataset = data.TensorDataset(tensor_x,tensor_y)
    train_loader = data.DataLoader(dataset=my_dataset, batch_size=BATCH_SIZE, shuffle=True)

    rnn = RNN()
    # print(rnn)
    
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
    loss_func = nn.BCELoss(reduction='none')                       # the target label is not one-hotted
    weight = torch.Tensor([0.01, 0.99])

    # training and testing
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):        # gives batch data
            b_x = b_x.view(-1, TIME_STEP, INPUT_SIZE)              # reshape x to (batch, time_step, input_size)
            # print(b_x)
            # print(b_y)
            weight_ = weight[b_y.data.view(-1).long()].view_as(b_y)
            output = rnn(b_x)                               # rnn output
            loss = loss_func(output, b_y)                   # cross entropy loss
            loss_class_weighted = loss * weight_
            loss_class_weighted = loss_class_weighted.mean()
            optimizer.zero_grad()                           # clear gradients for this training step
            loss_class_weighted.backward()                                 # backpropagation, compute gradients
            optimizer.step()                                # apply gradients

            if step % 30 == 0:               # (samples, time_step, input_size)
                pred_y = output.data.numpy()
                print(pred_y[:2])
                print(b_y[:2])
                pred_y[pred_y>0.5]=1
                pred_y[pred_y<=0.5]=-1
                b_y = b_y.numpy()
                hit = (pred_y == b_y)
                accuracy = float(hit.astype(int).sum()) / float(b_y.sum())
                print('Epoch: ', epoch, '| train loss: %.4f' % loss_class_weighted.data.numpy(), '| test accuracy: %.2f' % accuracy, '| hit num: ' % hit.astype(int).sum())
                print("\n")
