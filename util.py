import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import config as c

def load_data(path):
    with open(path) as f:
        l = list(filter(lambda r: r[1].startswith('Event'), enumerate(f.readlines())))
    events = np.array(l).T
    event_lines = events[0].astype(int)
    meta_lines = [0, 1]
    skip_rows = meta_lines + event_lines.tolist()
    df = pd.read_csv(path,
                     skiprows=skip_rows,
                     parse_dates=['Date'],
                     index_col='Date')
    df.drop(columns=['Unnamed: 12'], inplace=True)
    df = df.resample(f"{c.resample_mins}min", label="right").mean()
    df = df[c.cols]
    return df


def standard_scaler(df):
    standard_scaler = StandardScaler()
    df = pd.DataFrame(standard_scaler.fit_transform(df), index=df.index, columns=df.columns)
    return df


def remove_trainling_data(df):
    df = df.copy()
    diffs = df['RelativeDensity'] - df['RelativeDensity'].shift(1)
    last_drop = df[diffs.abs() > (diffs.mean() + 3*diffs.std())].index[-1]
    df['RelativeDensity'][:last_drop - 1*df.index.freq].plot()
    return df


def load_and_preprocess_train():
    training = load_data(c.dataset_path)
    train_bound = int(c.train_bound * len(training))
    train, val = training[:train_bound], training[train_bound:]
    train, val = standard_scaler(train), standard_scaler(val)
    val = remove_trainling_data(val)
    return train, val

def fxx(f, i):
    f_xx = (11*f[i-4]-56*f[i-3]+114*f[i-2]-104*f[i-1]+35*f[i+0])/(12)