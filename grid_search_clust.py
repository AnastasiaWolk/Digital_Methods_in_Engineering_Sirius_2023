import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm.contrib import tzip
from tqdm.notebook import tqdm

from sklearn.cluster import KMeans, Birch, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, mutual_info_score
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler

import warnings


def hyperparam_search(X: np.array, dist: np.array, model, hyperparams: dict):
    m = model(**hyperparams)
    y_hat = m.fit_predict(X)
    if np.unique(y_hat).size == 1:
        y_hat[0] = 2
    score_euclid = silhouette_score(X, y_hat, random_state=42, metric='euclidean')
    score_dtw = silhouette_score(dist, y_hat, random_state=42, metric='precomputed')
    return score_euclid, score_dtw

def grid_search(X: np.array, dist: dict, model, hyperparam_grid: ParameterGrid, verbose=True):
    '''No cv in sklearn since no labels.'''
    
    score_euclid_arr = np.array([])
    score_dtw_arr = np.array([])
    for param in tqdm(hyperparam_grid):
        score_euclid, score_dtw = hyperparam_search(X, dist, model, param)
        score_euclid_arr = np.append(score_euclid_arr, score_euclid)
        score_dtw_arr = np.append(score_dtw_arr, score_dtw)
    
    best_ind = np.argmax(score_euclid_arr)
    best_par = hyperparam_grid[best_ind]
    best_score_euclid = score_euclid_arr[best_ind]
    best_score_dtw = score_dtw_arr[best_ind]
    
    if verbose:
        plt.plot(range(len(score_euclid_arr)), score_euclid_arr, label='Euclidian')
        plt.plot(range(len(score_dtw_arr)), score_dtw_arr, label='DTW')
        plt.xlabel('Parameter set #')
        plt.ylabel('Silhouette score')
        plt.legend()
        plt.ylim(-1, 1)
        plt.show()
    
    return best_par, best_score_euclid, best_score_dtw

def grid_search_run(df:pd.DataFrame,
                    dist: np.array,
                    feat_indices: (list, np.array),
                    models_ls: list,
                    names_ls:list,
                    hyper_params_ls:list,
                    verbose=True):
    
    scaler = StandardScaler()
    X = df.iloc[:, feat_indices].values
    X_scaled = scaler.fit_transform(X)
    params_dict = {}
    
    for m, n, hp in tzip(models_ls, names_ls, hyper_params_ls):
        best_par, best_score_euclid, best_score_dtw = grid_search(X_scaled, dist, m, hp, verbose)
        params_dict[n] = [best_score_euclid, best_score_dtw, best_par]
        
        if verbose:
            print(f'>> {n}')
            print(f'>> Silhouette euclidian score = {best_score_euclid:.5f}')
            print(f'>> Silhouette dtw score = {best_score_dtw:.5f}')
        
    return params_dict