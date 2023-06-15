import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from dtaidistance import dtw#, similarity
from dtaidistance import dtw_visualisation as dtwvis
from functools import partial
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams['figure.figsize'] = (11, 7)
plt.rcParams['font.size'] = 20


def clear_data(df: pd.DataFrame):
    df['signal_len'] = df[['sp']].apply(lambda x: x[0].shape[0], axis=1)
    df = df.drop_duplicates(subset=['UWI'])
    return df

def compute_dtw_distance_wells(df: pd.DataFrame, dist_to_sim: bool, return_lower_tri:bool):
    start = time.time()
    alpha_ps_ls = df.loc[:, 'sp'].to_list()
    
    dist_arr = dtw.distance_matrix_fast(alpha_ps_ls)
    if dist_to_sim:
        raise ValueError('Not implemented')
        dist_arr = similarity.distance_to_similarity(dist_arr)
    
    if return_lower_tri:
        upper_tri_ind = np.triu_indices(len(alpha_ps_ls))
        dist_arr[upper_tri_ind] = -1
    
    delta = time.time() - start
    return dist_arr, delta


def get_spectrum(y: np.array, n: int, plot: bool):
    yf = np.fft.fft(y)
    xf = np.fft.fftfreq(len(y), 1)

    yf = (np.abs(yf) / len(y))
    yf = yf[xf >= 0]
    xf = xf[xf >= 0]
    spectrum, intensity = take_n_spectrum(n, yf, xf)
    
    if plot:
        plot_spectrum(xf, yf, ind)
    return *spectrum, *intensity

def plot_spectrum(xf:np.array, yf: np.array, name_well: str, xlim=[0,.5]):
    plt.plot(xf, yf)
    plt.xlabel('Freq')
    plt.ylabel('Intensity')
    
    plt.title(f'Well name `{name_well}`')
    plt.xlim(xlim)
    sp, ampl = take_n_spectrum(n, yf, xf)
    print('Highest frequencies and their aplitudes:', sp, ampl, sep='\n')
    
    
def take_n_spectrum(n: int, fft_arr: np.array, freq_arr: np.array):
    ind = np.argpartition(fft_arr, -n)[-n:]
    intensity = fft_arr[ind]
    spectrum = freq_arr[ind]
    return spectrum, intensity

def make_spectrum_columns(df: pd.DataFrame, n: int):
    sp_feat, ampl_feat = [], []
    for i in range(n):
        sp_name = "spect%d" % i
        ampl_name = "ampl%d" % i
        
        sp_feat.append(sp_name)
        ampl_feat.append(ampl_name)
        # df[sp_name] = np.nan
        # df_check[ampl_name] = np.nan
        
    return df, sp_feat+ampl_feat

def add_spectrum_features(df: pd.DataFrame, n: int):
    df, sp_ampl_feat = make_spectrum_columns(df, n)
    part_get_spectrum = partial(get_spectrum, n=n, plot=False)

    df[sp_ampl_feat] = np.vstack(df['sp'].apply(part_get_spectrum))
    return df

def plot_corr_matrix(df: pd.DataFrame, columns: list):
    df_local = df[columns]
    _, ax = plt.subplots(figsize=(11, 11))
    im = ax.imshow(df_local.corr())
    
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45)
    ax.set_yticks(range(len(columns)))
    ax.set_yticklabels(columns, rotation=45)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()
    return