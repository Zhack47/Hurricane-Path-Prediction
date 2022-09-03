# Hurricane-Path-Prediction

__Hurricane Path Prediction using a Recurring Neural Network__

In this repository you will find the code necessary to train a LSTM-based Neural Network to predict the next position of a hurricane.

## The Data

The data used was retrieved from the NOAA [HURDAT2](https://www.nhc.noaa.gov/data/#hurdat) database.

As explained on their website it contains the geographic progression data of each hurricane since 1851 (currently am using a version downloaded in 2020, only goes up to 2015)

The hurricanes' coordinates are available for the duration of the hurricane at 4  times in the day : 0000, 0600, 1200, 1800.

## The Model

I built the model using PyTorch.

It's a very shallow model, as I found out the resultss only worsened when I deepened the LSTM part.

- 1 Input Layer with Batch Normalization

- 1 LSTM Layer with 256 hidden features

- 1 Dense (Fuully Connected Layer)


## QUICKSTART

### Installation

```git clone https://github.com/Zhack47/Hurricane-Path-Prediction.git```

```cd Hurricane-Path-Prediction/```

```pip3 install -r requirements.txt```

### Training

```python3 train.py```

### Testing
```python3 test.py```
