# Python 3.7

import sys

from model import LSTM

import torch
import torch.nn as nn
import torch.optim
import pandas as pd

def load_data(stocks):
    stock_data = None
    for stock in stocks:
        this_stock = pd.read_csv("Stocks/" + stock + ".us.txt")
        this_stock = this_stock[["Date", "Close"]]
        if stock_data is None:
            stock_data = this_stock
        else:
            stock_data = pd.merge(stock_data, this_stock, on="Date")
    return stock_data.iloc[:,1:]

def train_stocks(stock_data, days_per_slice, batch_size, batches_per_epoch,
        model, epochs, learning_rate=0.001):
    # Sanity Checking
    known_days = stock_data.shape[0]
    num_features = stock_data.shape[1]
    total_days = days_per_slice * batch_size * batches_per_epoch
    assert total_days <= known_days
    assert days_per_slice > 2
    assert batch_size > 0
    assert batches_per_epoch > 0
    # Select the last total_days elements
    relevant_data = stock_data[known_days - total_days:]
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss = None
    for e in range(epochs):
        for batch_num in range(batches_per_epoch):
            # Select the current batch
            batch_data = relevant_data[
                     batch_num      * days_per_slice * batch_size :
                    (batch_num + 1) * days_per_slice * batch_size]
            batch = torch.tensor(batch_data.values).float()
            # Reshape it
            x = batch.view(batch_size, days_per_slice,
                    num_features).permute(1,0,2)
            # Slice off the last time stamp to use as the output
            x = x[:-1]
            y = x[-1]
            # Optimize
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
    return model, loss

if __name__ == "__main__":
    train_all = len(sys.argv) > 1
    # Parameters
    days_per_slice = 14
    batch_size = 32
    max_epochs = 5000
    epoch_steps = 100
    epoch_gap = max_epochs // epoch_steps
    folds = 5
    # Automobile stocks: Toyota, Ford, Honda, Tata (lol)
    pd_data = load_data(["tm", "f", "hmc", "ttm"]).pct_change()[1:]
    known_days = pd_data.shape[0]
    num_features = pd_data.shape[1]
    all_results = []
    loss_fn = nn.MSELoss()
    if not train_all:
        for fold in range(folds):
            results = pd.DataFrame()
            model = LSTM(input_size=pd_data.shape[1], hidden_size=32,
                    num_layers=4, dropout=0.5)
            for step in range(epoch_steps):
                # Select Testing Fold
                test_start = fold * (known_days // folds)
                if fold == folds - 1:
                    test_end = known_days
                else:
                    test_end = (fold + 1) * (known_days // folds)
                test_data = pd_data[test_start:test_end]
                # Train the region before/after the testing fold individually
                train_data_prior = None
                train_data_post = None
                if not fold == folds - 1:
                    train_data_post = pd_data[test_end:]
                if not fold == 0:
                    train_data_prior = pd_data[:test_start]
                # Train
                if train_data_prior is not None:
                    model, loss_train = train_stocks(
                            train_data_prior,
                            days_per_slice,
                            batch_size,
                            train_data_prior.shape[0] // (days_per_slice
                                                              * batch_size),
                            model,
                            epoch_gap)
                if train_data_post is not None:
                    model, loss_train = train_stocks(
                            train_data_post,
                            days_per_slice,
                            batch_size,
                            train_data_post.shape[0] // (days_per_slice
                                                              * batch_size),
                            model,
                            epoch_gap)
                # Calculate loss on test set as one large batch
                test_batch_size = test_data.shape[0] // days_per_slice
                test = torch.tensor(
                        test_data[:test_batch_size * days_per_slice].values).float()
                x_test = test.view(test_batch_size, days_per_slice,
                        num_features).permute(1,0,2)
                x_test = x_test[:-1]
                y_test = x_test[-1]
                test_pred = model(x_test)
                loss_test = loss_fn(test_pred, y_test)
                print("Fold {}".format(fold),
                        "Total Epochs = {}".format((step + 1) * epoch_gap),
                        "Loss = {}".format(loss_test))
                print("(First batch) Truth:", y_test[0].flatten().tolist(),
                        "Prediction:", test_pred[0].flatten().tolist())
                # Record results
                results.insert(step, step * epoch_gap, [loss_train.item(), loss_test.item()])
            all_results.append(results)
        # Write results to disk
        for fold in range(folds):
            all_results[fold].to_csv("fold-{}-results.csv".format(fold))
    else: # train_all
        model = LSTM(input_size=pd_data.shape[1], hidden_size=32,
                num_layers=4, dropout=0.5)
        pd_data = pd_data.iloc[:len(pd_data)-365] # don't train on last year
        model, loss_train = train_stocks(
                pd_data, 
                days_per_slice,
                batch_size,
                pd_data.shape[0] // (days_per_slice * batch_size),
                model,
                500)
        print("Trained on entire dataset for 500 epochs.")
        print("Training set loss:", loss_train.item())
        torch.save(model.state_dict(), "model.pt")
