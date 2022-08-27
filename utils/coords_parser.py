import csv

import numpy as np
import torch


def intify_atlante(data):
    if 'W' in data or 'S' in data:
        return -float(data[:-1])
    elif 'E' in data or 'N' in data:
        return float(data[:-1])


def intify_pacifique(data):
    if 'W' in data or 'S' in data:
        return 360-float(data[:-1])
    elif 'E' in data or 'N' in data:
        return float(data[:-1])


def intify(data, ocean):
    if 'pacific' in ocean:
        return intify_pacifique(data)
    else:
        return intify_atlante(data)


def coords_parser(filename):
    """
    Parses NOAA csv file to extract the coordinates of the hurricane/ storm during its progression.

    :param filename: Path to the csv file
    :return A dictionnary with athe Hurricane ID as key, and a list of its coordinates.
    Each coordinate is a list of two float values
    """
    coords_dict = {}
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            _key = row['ID']
            value_lon = (intify(row['Longitude'], filename)+65) /90
            value_lat = (intify(row['Latitude'], filename)-27) /45
            try:
                coords_dict[_key].append([value_lat, value_lon])
            except KeyError:
                coords_dict[_key] = [[value_lat, value_lon]]
    count = 0
    mean = 0
    for key in coords_dict:
        # count += len(coords_dict[key])
        # mean += np.sum(coords_dict[key], axis=0)
        coords_dict[key] = torch.tensor(np.array(coords_dict[key]))
    # print(f"Mean : {mean/count}")
    return coords_dict

