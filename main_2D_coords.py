import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
from torch.nn import MSELoss, BCELoss
from torch.optim import Adam
from tqdm import tqdm

from utils.preprocessing import train_val_test_split, remove_small_samples
from utils.torch_rnn import HurricaneRNN
from utils.coords_parser import coords_parser
import numpy as np



def train_pacific_lon(size: int = 10, nb_epochs: int = 10, batch_size: int = 16):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    model = HurricaneRNN("Atlantic", size-1)
    mse = MSELoss()
    opt = Adam(params=model.parameters(), lr=1e-3)
    dict_atlantique = coords_parser("Data/atlantic.csv")
    dict_atlantique = remove_small_samples(dict_atlantique, min_size=size)
    train_keys, val_keys, test_keys = train_val_test_split(list(dict_atlantique.keys()))
    x_train = [dict_atlantique[key][:size-1] for key in train_keys]
    y_train = [dict_atlantique[key][size-1:size] for key in train_keys]
    x_val = [dict_atlantique[key][:size-1] for key in val_keys]
    y_val = [dict_atlantique[key][size-1:size] for key in val_keys]

    nb_data_train = len(x_train)
    nb_data_val = len(x_val)

    if not (nb_data_train == len(y_train)):
        raise ValueError(f" X and Y must be of same length. Found x : {nb_data_train} and y : {len(y_train)}")

    nb_iters = nb_data_train//batch_size
    nb_iters_val = nb_data_val//batch_size

    history = {}
    history["train"] = []
    history["validation"] = []
    for epoch in tqdm(range(nb_epochs)):
        mean_loss = 0.0
        mean_loss_val = 0.0
        for i in range(nb_iters):
            data = torch.stack(x_train[i*batch_size:(i+1)*batch_size], dim=0).float()  # .to(device)
            gt = torch.stack(y_train[i*batch_size:(i+1)*batch_size], dim=0).float()
            out = model(data)
            loss = torch.sqrt(mse(torch.unsqueeze(out, dim=1), gt))
            opt.zero_grad()
            loss.backward()
            opt.step()
            mean_loss+=loss.detach()

        for i_val in range(nb_iters_val):
            data = torch.stack(x_val[i_val*batch_size:(i_val+1)*batch_size], dim=0).float()  # .to(device)
            gt = torch.stack(y_val[i_val*batch_size:(i_val+1)*batch_size], dim=0).float()
            out = model(data)
            loss = torch.sqrt(mse(torch.unsqueeze(out, dim=1), gt))
            mean_loss_val += loss.detach()

        print(f"Training Loss : {mean_loss/nb_iters:.3}")
        print(f"Validation Loss : {mean_loss_val/nb_iters_val:.3}")
        history["train"].append(mean_loss/nb_iters)
        history["validation"].append(mean_loss_val/nb_iters_val)

    torch.save(model.state_dict(), f"models/HurricaneRes_RNN_2D_Atlantic_1LSTMS_256_{device}.h5")
    return history


def test_pacific_lon(size, batch_size=16):
    model = HurricaneRNN("Atlantic", size-1, 0.2)
    model.load_state_dict(torch.load("models/HurricaneRes_RNN_2D_Atlantic_1LSTMS_256_cpu.h5"))

    dict_atlantique = coords_parser("Data/atlantic.csv")
    dict_atlantique = remove_small_samples(dict_atlantique, min_size=size)

    train_keys, val_keys, test_keys = train_val_test_split(list(dict_atlantique.keys()))
    x_test = [dict_atlantique[key][:size-1] for key in test_keys]
    y_test = [dict_atlantique[key][size-1:size] for key in test_keys]
    nb_data_test = len(x_test)

    if not nb_data_test == len(y_test):
        raise ValueError(f" X and Y must be of same length. Found x : {nb_data_train} and y : {len(pac_lon_y_test)}")

    nb_iters_test = nb_data_test // batch_size
    mean_loss = 0.0
    mse = MSELoss()

    result = []
    for i in tqdm(range(nb_iters_test)):
        data = torch.stack(x_test[i * batch_size:(i + 1) * batch_size], dim=0).float()  # .to(device)
        gt = torch.stack(y_test[i * batch_size:(i + 1) * batch_size], dim=0).float()
        out = model(data)

        loss = torch.sqrt(mse(torch.unsqueeze(out, dim=1), gt))
        mean_loss += loss.detach()
        result.append((x_test[i * batch_size:(i + 1) * batch_size], y_test[i * batch_size:(i + 1) * batch_size], out.detach().numpy()))
    print(f"Test Loss {mean_loss / nb_iters_test}")
    return result


history = train_pacific_lon(size=10, nb_epochs=60, batch_size=16)

points = test_pacific_lon(size=10, batch_size=16)

plt.plot(history["train"])
plt.plot(history["validation"])
plt.show()

for batch in points:
    x, y, y_hat = batch
    for i in range(len(x)):
        for vec in x[i].detach().numpy():
            plt.scatter(vec[1]*90, vec[0]*45, c="blue")
        plt.scatter(y[i].detach().numpy()[0][1]*90, y[i].detach().numpy()[0][0]*45, c='red')
        plt.scatter(y_hat[i][1]*90, y_hat[i][0]*45, c='pink')
    plt.show()
