import torch
from matplotlib import pyplot as plt
from torch.nn import MSELoss
from tqdm import tqdm

from utils.coords_parser import coords_parser
from utils.preprocessing import train_val_test_split, remove_small_samples
from utils.torch_rnn import HurricaneRNN


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
        raise ValueError(f" X and Y must be of same length. Found x : {nb_data_test} and y : {len(y_test)}")

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

if __name__ == "__main__":
    points = test_pacific_lon(size=10, batch_size=16)

    for batch in points:
        x, y, y_hat = batch
        for i in range(len(x)):
            for vec in x[i].detach().numpy():
                plt.scatter(vec[1] * 90, vec[0] * 45, c="blue")
            plt.scatter(y[i].detach().numpy()[0][1] * 90, y[i].detach().numpy()[0][0] * 45, c='red')
            plt.scatter(y_hat[i][1] * 90, y_hat[i][0] * 45, c='pink')
        plt.show()
