import torch
from matplotlib import pyplot as plt
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import tqdm

from utils.coords_parser import coords_parser
from utils.preprocessing import remove_small_samples, train_val_test_split
from utils.torch_rnn import HurricaneRNN


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

    torch.save(model.state_dict(), f"models/HurricaneRes_RNN_2D_Atlantic_1LSTMS_256_{device}.h5")
    return history


if __name__ == "__main__":

    history = train_pacific_lon(size=10, nb_epochs=60, batch_size=16)

    plt.plot(history["train"])
    plt.plot(history["validation"])
    plt.show()