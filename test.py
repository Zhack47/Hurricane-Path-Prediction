import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy.ma import concatenate
from numpy.random import shuffle
from paramiko.py3compat import long
from torch.nn import MSELoss
from torchmetrics import MeanAbsolutePercentageError, MeanAbsoluteError
from tqdm import tqdm

from utils.coords_parser import coords_parser
from utils.preprocessing import train_val_test_split, remove_small_samples
from utils.torch_rnn import HurricaneRNN
import geopy.distance

def test_pacific_lon(size, batch_size=16):
    model = HurricaneRNN("Atlantic", size-1, 0.2)

    model.load_state_dict(torch.load("models/HurricaneRes_RNN_2D_Atlantic_1LSTMS_256_cpu.h5"))
    #model.load_state_dict(torch.load("models/HurricaneRes_RNN_2D_Atlantic_1LSTMS_256_cpu_short_term_5pts.h5"))

    dict_atlantique = coords_parser("Data/atlantic.csv")
    dict_atlantique = remove_small_samples(dict_atlantique, min_size=size)

    train_keys, val_keys, test_keys = train_val_test_split(list(dict_atlantique.keys()))
    x_test = []
    y_test = []
    for key in test_keys:
        for i in range(len(dict_atlantique[key])-size+1):
            x_test.append(dict_atlantique[key][i:size-1+i])
            y_test.append(dict_atlantique[key][size - 1 + i:size + i])
    nb_data_test = len(x_test)
    permut = list(range(nb_data_test))
    shuffle(permut)
    x_test_, y_test_ = [x_test[i] for i in permut], [y_test[i] for i in permut]
    x_test, y_test = x_test_, y_test_
    if not nb_data_test == len(y_test):
        raise ValueError(f" X and Y must be of same length. Found x : {nb_data_test} and y : {len(y_test)}")

    nb_iters_test = nb_data_test // batch_size
    mean_rmse = 0.0
    mean_mae= 0.0
    mean_dist= 0.0
    rmse = MSELoss()
    mae = MeanAbsoluteError()

    result = []
    for i in tqdm(range(nb_iters_test)):
        data = torch.stack(x_test[i * batch_size:(i + 1) * batch_size], dim=0).float()  # .to(device)
        gt = torch.stack(y_test[i * batch_size:(i + 1) * batch_size], dim=0).float()
        out = model(data)
        out= torch.add(torch.multiply(out, torch.tensor([[10, 20]])), torch.tensor([[27,-65]]))
        gt= torch.add(torch.multiply(gt, torch.tensor([[[10, 20]]])), torch.tensor([[[27,-65]]]))
        rmse_value = torch.sqrt(rmse(torch.unsqueeze(out, dim=1), gt))
        mae_value = mae(torch.unsqueeze(out, dim=1), gt)
        mean_rmse += rmse_value.detach()
        mean_mae += mae_value.detach()
        lat_dist = torch.mean(59.9 * (out[:,0] - gt[:,:,0]))
        lon_dist = torch.mean(47.79 * (out[:,0] - gt[:,:,0]))
        for i_ in range(batch_size):
            lat1 = out[i_, 0].detach().numpy()
            lon1 = out[i_, 1].detach().numpy()
            lat2 = gt[i_, 0, 0].detach().numpy()
            lon2 = gt[i_, 0, 1].detach().numpy()
            mean_dist+= geopy.distance.geodesic((lat1, lon1), (lat2, lon2)).nautical
            print(mean_dist)
        result.append((x_test[i * batch_size:(i + 1) * batch_size], gt, out.detach().numpy()))
    print(f"Test RMSE {mean_rmse / nb_iters_test}")
    print(f"Test MSE {(mean_rmse / nb_iters_test)**2}")
    print(f"Test MAE {mean_mae / nb_iters_test}")
    print(f"Test Distance (approx.):  {mean_dist / (nb_iters_test * batch_size)} n miles")
    return result

if __name__ == "__main__":
    points = test_pacific_lon(size=10, batch_size=16)

    for batch in points:
        x, y, y_hat = batch
        for i in range(len(x)):
            x[i] = torch.add(torch.multiply(x[i], torch.tensor([[10, 20]])), torch.tensor([[27, -65]]))
            for vec in x[i].detach().numpy():
                plt.scatter(vec[1] , vec[0] * 1, c="blue")
            plt.scatter(y[i].detach().numpy()[0][1], y[i].detach().numpy()[0][0], c='red')
            plt.scatter(y_hat[i][1], y_hat[i][0], c='pink')
        plt.show()
