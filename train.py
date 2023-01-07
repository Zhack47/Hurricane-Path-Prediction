import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy.random import shuffle
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import tqdm

from utils.coords_parser import coords_parser
from utils.preprocessing import remove_small_samples, train_val_test_split
from utils.torch_rnn import HurricaneRNN


def train_pacific_lon(size: int = 10, nb_epochs: int = 10, batch_size: int = 16, model_h5_filename="models/HurricaneRes_RNN_2D_Atlantic_1LSTMS_256_cpu.h5"):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    model = HurricaneRNN("Atlantic", size-1, dropout=0.35)
    mse = MSELoss()
    opt = Adam(params=model.parameters(), lr=1e-4, weight_decay=1e-4)
    dict_atlantique = coords_parser("Data/atlantic.csv")
    dict_atlantique = remove_small_samples(dict_atlantique, min_size=size)
    train_keys, val_keys, test_keys = train_val_test_split(list(dict_atlantique.keys()))
    x_train = []
    x_val = []
    y_train = []
    y_val = []
    for key in train_keys:
        for i in range(len(dict_atlantique[key])-size+1):
            x_train.append(dict_atlantique[key][i:size-1+i])
            y_train.append(dict_atlantique[key][size - 1 + i:size+i])
    for key in val_keys:
        for i in range(len(dict_atlantique[key])-size+1):
            x_val.append(dict_atlantique[key][i:size-1+i])
            y_val.append(dict_atlantique[key][size - 1 + i:size + i])
    print(np.shape((x_train)))

    nb_data_train = len(x_train)
    nb_data_val = len(x_val)

    permut = list(range(nb_data_train))
    shuffle(permut)
    x_train_, y_train_ = [x_train[i] for i in permut], [y_train[i] for i in permut]
    x_train, y_train = x_train_, y_train_


    permut = list(range(nb_data_val))
    shuffle(permut)
    x_val_, y_val_ = [x_val[i] for i in permut], [y_val[i] for i in permut]
    x_val, y_val = x_val_, y_val_

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
            mean_loss += loss.detach()

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

    torch.save(model.state_dict(), model_h5_filename)
    return history


if __name__ == "__main__":
    previous_data_points = 10
    history = train_pacific_lon(size=5, nb_epochs=120, batch_size=16, model_h5_filename="models/HurricaneRes_RNN_2D_Atlantic_1LSTMS_256_cpu_short_term_5pts.h5")

    plt.plot(history["train"])
    plt.plot(history["validation"])
    plt.legend("")
    plt.show()
