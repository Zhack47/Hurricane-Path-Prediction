def train_val_test_split(data, test_split: float = 1/6, val_split: float = 1/3):
    len_data = len(data)
    test_idx_start = round((1-test_split)*len_data)
    val_idx_start = round((1-val_split)*test_idx_start)
    return data[:val_idx_start], data[val_idx_start:test_idx_start], data[test_idx_start:]


def remove_small_samples(dictionnary, min_size: int = 10):
    keys = list(dictionnary.keys())
    for _key in keys:
        if len(dictionnary[_key]) < min_size:
            dictionnary.pop(_key)
    return dictionnary
