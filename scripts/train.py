""" Module for model training. """


import yaml
import argparse
import math
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import preprocess


with open("../model/params.yaml", "r") as params_file:
    params = yaml.safe_load(params_file)

model_dir = params['model_dir']


class TimeSeriesDataset(Dataset):   
    def __init__(self, X, y, seq_len=1, step=1):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.step = step

    def __len__(self):
        return self.X.__len__() - self.seq_len - self.step

    def __getitem__(self, index):
        return self.X[index:index+self.seq_len], self.y[index+self.seq_len+self.step-1]


class TSModel(nn.Module):
    def __init__(self, n_features, n_hidden=64, n_layers=1):
        super(TSModel, self).__init__()

        self.n_hidden = n_hidden
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            batch_first=True,
            num_layers=n_layers,
            dropout=0.5
        )
        self.linear = nn.Linear(n_hidden, 1)
        self.dropout = nn.Dropout(0.5)
        
    # lstm_output: (batch_size, seq_len, n_hidden)
    # final_state: (1, batch_size, n_hidden)    
    def attention(self, lstm_output, final_state):
        final_state = final_state.permute(1, 2, 0) # (1,b,h)->(b,h,1)
        weights = torch.bmm(lstm_output, final_state)  # (b,s,1)
        weights = F.softmax(weights, dim=1)

        return torch.bmm(lstm_output.permute(0, 2, 1), weights).squeeze(2) # (b,h)

    def forward(self, x):
        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden * num_directions]
        # output : [batch_size, seq_len, n_hidden * num_directions]
        output, (hn, cn) = self.lstm(x)

        attn_output = self.attention(output, hn)
        y_pred = self.linear(attn_output)
        return y_pred
    

def train_model(
        train_df,
        test_df,
        label_name,
        sequence_length,
        step,
        batch_size,
        n_epochs,
        n_epochs_stop
):
    """Train LSTM model."""
    print("Starting with model training...")

    # create dataloaders
    train_dataset = TimeSeriesDataset(np.array(train_df), np.array(train_df[label_name]), seq_len=sequence_length, step=step)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TimeSeriesDataset(np.array(test_df), np.array(test_df[label_name]), seq_len=sequence_length, step=step)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    # set up training
    n_features = train_df.shape[1]
    model = TSModel(n_features)
    criterion = torch.nn.MSELoss()  # L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # export model for visualization
    # input_names = ['Sentence']
    # output_names = ['yhat']
    # dummy_input = torch.randn(1, 1, n_features)
    # torch.onnx.export(model, dummy_input, '../model/rnn.onnx', input_names=input_names, output_names=output_names)

    train_hist = []
    test_hist = []

    # start training
    best_loss = np.inf
    epochs_no_improve = 0
    for epoch in range(1, n_epochs+1):
        running_loss = 0
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader, 1):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            data = torch.Tensor(np.array(data))
            output = model(data)
            loss = criterion(output.flatten(), target.type_as(output))
            # if type(criterion) == torch.nn.modules.loss.MSELoss:
            #     loss = torch.sqrt(loss)  # RMSE
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        running_loss /= len(train_loader)
        train_hist.append(running_loss)

        # test loss
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = torch.Tensor(np.array(data))
                output = model(data)
                loss = criterion(output.flatten(), target.type_as(output))
                test_loss += loss.item()
            test_loss /= len(test_loader)
            test_hist.append(test_loss)

            # early stopping
            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(model.state_dict(), Path(model_dir, 'model.pt'))
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve == n_epochs_stop:
                print("Early stopping.")
                break

        print(f'Epoch {epoch} train loss: {round(running_loss,4)} test loss: {round(test_loss,4)}')

        hist = pd.DataFrame()
        hist['training_loss'] = train_hist
        hist['test_loss'] = test_hist

    print("Completed.")

    return hist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=params['sequence_length'])
    parser.add_argument("--step", type=int, default=params['step'])
    parser.add_argument("--batch-size", type=int, default=params['batch_size'])
    parser.add_argument("--n-epochs", type=int, default=params['n_epochs'])
    parser.add_argument("--n-epochs-stop", type=int, default=params['n_epochs_stop'])
    args = parser.parse_args()

    train_df = preprocess.load_data('train.csv')
    test_df = preprocess.load_data('test.csv')
    label_name = params['label_name']

    train_model(
        train_df,
        test_df,
        label_name,
        args.sequence_length,
        args.step,
        args.batch_size,
        args.n_epochs,
        args.n_epochs_stop
    )
