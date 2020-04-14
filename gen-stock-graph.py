# Python 3.7

from model import LSTM
from main import load_data

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def reconstruct_price_prediction(prices, movement_prediction):
    prediction = movement_prediction.tolist()
    assert len(prices) == len(movement_prediction)
    ret = []
    for i in range(len(prices)):
        ret.append(prices[i]+prices[i]*movement_prediction[i])
    return ret

if __name__ == "__main__":
    pd_data = load_data(["tm", "f", "hmc", "ttm"]).pct_change()[1:] 
    num_features = pd_data.shape[1]
    model = LSTM(input_size=num_features, hidden_size=32,
            num_layers=4, dropout=0.5)
    model.load_state_dict(torch.load("model.pt"))
    
    tm_price_changes_col  = pd_data.iloc[:,0]
    f_price_changes_col   = pd_data.iloc[:,1]
    hmc_price_changes_col = pd_data.iloc[:,2]
    ttm_price_changes_col = pd_data.iloc[:,3]
    
    tm_price_changes  = tm_price_changes_col.to_list()
    f_price_changes   = f_price_changes_col.to_list()
    hmc_price_changes = hmc_price_changes_col.to_list()
    ttm_price_changes = ttm_price_changes_col.to_list()
    
    days_per_slice = 14
    pd_last_year = pd_data.iloc[len(pd_data) - 365:,:]
    windows = 365 - days_per_slice + 1
    windows_data = []
    for i in range(windows):
        data_slice = pd_last_year.iloc[i : i + days_per_slice, :]
        windows_data.append(data_slice)
    pd_predict = pd.concat(windows_data)
    
    predict_batch_size = pd_predict.shape[0] // days_per_slice
    predict = torch.tensor(
            pd_predict[:predict_batch_size
                           * days_per_slice].values).float()
    x_predict = predict.view(predict_batch_size, days_per_slice,
            num_features).permute(1,0,2)
    y_predict = model(x_predict).detach().numpy()

    tm_price_changes_year  = tm_price_changes[len(tm_price_changes) - 365:]
    f_price_changes_year   = f_price_changes[len(f_price_changes) - 365:]
    hmc_price_changes_year = hmc_price_changes[len(hmc_price_changes) - 365:]
    ttm_price_changes_year = ttm_price_changes[len(ttm_price_changes) - 365:]
    
    tm_predict_price_changes = y_predict[:,0]
    f_predict_price_changes = y_predict[:,0]
    hmc_predict_price_changes = y_predict[:,0]
    ttm_predict_price_changes = y_predict[:,0]
    tm_predict_padded = np.insert(tm_predict_price_changes, 0, np.zeros(days_per_slice))
    f_predict_padded = np.insert(f_predict_price_changes, 0, np.zeros(days_per_slice))
    hmc_predict_padded = np.insert(hmc_predict_price_changes, 0, np.zeros(days_per_slice))
    ttm_predict_padded = np.insert(ttm_predict_price_changes, 0, np.zeros(days_per_slice))

    pd_true_prices = load_data(["tm", "f", "hmc", "ttm"])
    tm_price = pd_true_prices.iloc[:,0].tolist()
    f_price = pd_true_prices.iloc[:,1].tolist()
    hmc_price = pd_true_prices.iloc[:,2].tolist()
    ttm_price = pd_true_prices.iloc[:,3].tolist()

    tm_price = tm_price[len(tm_price) - 365 + days_per_slice - 1 :]
    f_price = f_price[len(f_price) - 365 + days_per_slice - 1 :]
    hmc_price = hmc_price[len(hmc_price) - 365 + days_per_slice - 1 :]
    ttm_price = ttm_price[len(ttm_price) - 365 + days_per_slice - 1 :]
    
    tm_pred_price = reconstruct_price_prediction(tm_price, tm_predict_price_changes)
    f_pred_price = reconstruct_price_prediction(f_price, f_predict_price_changes)
    hmc_pred_price = reconstruct_price_prediction(hmc_price, hmc_predict_price_changes)
    ttm_pred_price = reconstruct_price_prediction(ttm_price, ttm_predict_price_changes)

    plt.figure(dpi=300)
    plt.plot(range(len(tm_price)), tm_price, "k", linewidth=0.5, label="Price")
    plt.plot([t + 1 for t in range(len(tm_price))], tm_pred_price, "b",
            linewidth=0.5, label="Prediction")
    plt.xlabel("Day")
    plt.ylabel("Price (USD)")
    plt.title("Predicted versus Actual Price: Toyota")
    plt.legend()
    plt.tight_layout()
    plt.savefig("tm_predict_price.png")

    plt.figure(dpi=300)
    plt.plot(range(len(f_price)), f_price, "k", linewidth=0.5, label="Price")
    plt.plot([t + 1 for t in range(len(f_price))], f_pred_price, "b",
            linewidth=0.5, label="Prediction")
    plt.xlabel("Day")
    plt.ylabel("Price (USD)")
    plt.title("Predicted versus Actual Price: Ford")
    plt.legend()
    plt.tight_layout()
    plt.savefig("f_predict_price.png")

    plt.figure(dpi=300)
    plt.plot(range(len(hmc_price)), hmc_price, "k", linewidth=0.5, label="Price")
    plt.plot([t + 1 for t in range(len(hmc_price))], hmc_pred_price, "b",
            linewidth=0.5, label="Prediction")
    plt.xlabel("Day")
    plt.ylabel("Price (USD)")
    plt.title("Predicted versus Actual Price: Honda")
    plt.legend()
    plt.tight_layout()
    plt.savefig("hmc_predict_price.png")

    plt.figure(dpi=300)
    plt.plot(range(len(ttm_price)), ttm_price, "k", linewidth=0.5, label="Price")
    plt.plot([t + 1 for t in range(len(ttm_price))], ttm_pred_price, "b",
            linewidth=0.5, label="Prediction")
    plt.xlabel("Day")
    plt.ylabel("Price (USD)")
    plt.title("Predicted versus Actual Price: Tata")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ttm_predict_price.png")

