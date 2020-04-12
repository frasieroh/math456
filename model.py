# Python 3.7

import sys

import torch
import torch.nn as nn
import torch.optim

# Adapted from https://www.jessicayung.com/lstms-for-time-series-in-pytorch/

class LSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size=None):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        if output_size is None:
            self.output_size = input_size
        else:
            self.output_size = output_size
        
        self.lstm = nn.LSTM(self.input_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def init_hidden(self, batch_size):
        return (torch.randn(1, batch_size, self.hidden_size),
                torch.randn(1, batch_size, self.hidden_size))

    def forward(self, x):
        batch_size = x.size()[1]
        hidden = self.init_hidden(batch_size)
        output, hidden = self.lstm(x, hidden)
        return self.output_layer(output[-1])

if __name__ == "__main__":
    print("Self-test with fake data")
    INPUT_SIZE  = 16 
    HIDDEN_SIZE = 16
    TIME_SLICE  = 16
    # Initialize model
    model = LSTM(INPUT_SIZE, HIDDEN_SIZE, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Generate fake data
        x = torch.zeros(TIME_SLICE, 1, INPUT_SIZE)
        y = torch.zeros(1, INPUT_SIZE)
        for t in range(TIME_SLICE):
            for i in range(INPUT_SIZE):
                x[t][0][i] = t + i
        for i in range(INPUT_SIZE):
            y[0][i] = TIME_SLICE + i
        for t in range(epochs): # Epochs
            model.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            print("Epoch:", t, "MSE:", loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Final prediction:")
        print(y_pred)
