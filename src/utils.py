import pandas as pd

from config import raw_data_path


def load_data(path=None):
    path = path if path else raw_data_path
    df = pd.read_csv(path, low_memory=False)
    return df