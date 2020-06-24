from scipy.spatial.distance import cityblock
from scipy.spatial.distance import euclidean
from sklearn.metrics import calinski_harabaz_score, silhouette_score, adjusted_mutual_info_score, adjusted_rand_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def intra_to_inter(X, y, dist, r):
    """Compute intracluster to intercluster distance ratio

    Parameters
    ----------
    X : array
        Design matrix with each row corresponding to a point
    y : array
        Class label of each point
    dist : callable
        Distance between two points. It should accept two arrays, each
        corresponding to the coordinates of each point
    r : integer
        Number of pairs to sample

    Returns
    -------
    ratio : float
        Intracluster to intercluster distance ratio
    """
    random_choices = []
    numerators = []
    denominators = []

    np.random.seed(11)

    for i in range(r):
        random_choices.append(np.random.randint(0, len(X), 2))

    for i in range(len(random_choices)):
        if random_choices[i][0] == random_choices[i][1]:
            continue

        elif y[random_choices[i][0]] == y[random_choices[i][1]]:
            numerators.append(dist(X[random_choices[i][0]],
                                        X[random_choices[i][1]]))
        else:
            denominators.append(dist(X[random_choices[i][0]],
                                        X[random_choices[i][1]]))

    return (np.asarray(numerators).mean()) / (np.asarray(denominators).mean())

def cluster_range(X, clusterer, k_start, k_stop, actual=None):
    
    chs = []
    iidrs = []
    inertias = []
    scs = []
    ys = []
    amis = []
    ars = []
    ps = []
    
    for i in range(k_start, k_stop+1):
        
        clusterer2 = clusterer
        clusterer2.n_clusters = i
        ys.append(clusterer2.fit_predict(X))

        iidrs.append(intra_to_inter(X, ys[-1], euclidean, 50))
        chs.append(calinski_harabaz_score(X, ys[-1]))
        inertias.append(clusterer2.inertia_)
        scs.append(silhouette_score(X, ys[-1]))

    keys = ['ys', 'iidrs', 'chs', 'inertias', 'scs']
    values = [ys, iidrs, chs, inertias, scs]

    if actual is not None:
        
        for i in ys:
            ps.append(purity(actual, i))
            ars.append(adjusted_rand_score(actual, i))
            amis.append(adjusted_mutual_info_score(actual, i))
            
        keys.extend(['ps', 'ars', 'amis'])
        values.append(ps)
        values.append(ars)
        values.append(amis)
        
        return dict(zip(keys, values))

    else:
        
        return dict(zip(keys, values))
    
def plot_internal(inertias, chs, iidrs, scs):
    """Plot internal validation values"""
    fig, ax = plt.subplots(nrows=2, ncols=2, dpi=200)
    ks = np.arange(2, len(inertias)+2)
    ax[0,0].plot(ks, inertias, '-o', label='SSE')
    ax[0,1].plot(ks, chs, '-ro', label='CH')
    ax[0,0].set_xlabel('$k$')
    ax[0,0].set_ylabel('SSE')
    ax[0,1].set_ylabel('CH')
#     lines, labels = ax.get_legend_handles_labels()
#     ax2 = ax.twinx()
    ax[1,0].plot(ks, iidrs, '-go', label='Inter-intra')
    ax[1,1].plot(ks, scs, '-ko', label='Silhouette coefficient')
    ax[1,0].set_ylabel('Inter-Intra')
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax2.legend(lines+lines2, labels+labels2)
    ax[1,1].set_ylabel('Silhouette Score')
    fig.tight_layout()
    return fig

def plot_clusters(X, ys):
    """Plot clusters given the design matrix and cluster labels"""
    k_max = len(ys) + 1
    k_mid = k_max//2 + 2
    fig, ax = plt.subplots(2, k_max//2, dpi=150, sharex=True, sharey=True, 
                           figsize=(7,4), subplot_kw=dict(aspect='equal'),
                           gridspec_kw=dict(wspace=0.01))
    for k,y in zip(range(2, k_max+1), ys):
        if k < k_mid:
            ax[0][k%k_mid-2].scatter(*zip(*X), c=y, s=1, alpha=0.8)
            ax[0][k%k_mid-2].set_title('$k=%d$'%k)
        else:
            ax[1][k%k_mid].scatter(*zip(*X), c=y, s=1, alpha=0.8)
            ax[1][k%k_mid].set_title('$k=%d$'%k)
    return ax