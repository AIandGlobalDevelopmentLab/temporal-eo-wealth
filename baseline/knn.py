import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sklearn.model_selection

def train_knn_logo(dists, features, labels, group_labels, cv_groups, test_groups,
                       weights=None, plot=True, group_names=None):
    '''Leave-one-group-out cross-validated training of a KNN model.
    Args
    - dists: np.array, shape [N, N], precomputed distance matrix
    - features: np.array, shape [N, D]
    - labels: np.array, shape [N]
    - group_labels: np.array, shape [N], type np.int32
    - cv_groups: list of int, labels of groups to use for LOGO-CV
    - test_groups: list of int, labels of groups to test on
    - weights: np.array, shape [N]
    - plot: bool, whether to plot MSE as a function of k
    - group_names: list of str, names of the groups, only used when plotting
    Returns
    - test_preds: np.array, predictions on indices from test_groups
    '''
    cv_indices = np.isin(group_labels, cv_groups).nonzero()[0]
    test_indices = np.isin(group_labels, test_groups).nonzero()[0]

    dists_cv = dists[np.ix_(cv_indices, cv_indices)]
    X = features[cv_indices]
    y = labels[cv_indices]
    groups = group_labels[cv_indices]
    w = None if weights is None else weights[cv_indices]

    ks = 2 ** np.arange(0, 11)  # 1 to 1024
    preds = np.ones([len(ks), len(cv_indices)], dtype=np.float64) * np.nan
    group_mses = np.ones([len(ks), len(cv_groups)], dtype=np.float64) * np.nan
    leftout_group_labels = np.zeros(len(cv_groups), dtype=groups.dtype)
    logo = sklearn.model_selection.LeaveOneGroupOut()

    for g, (train_indices, val_indices) in enumerate(logo.split(X, groups=groups)):
        leftout_group_labels[g] = groups[val_indices[0]]

        train_X, train_y = X[train_indices], y[train_indices]
        val_y = y[val_indices]
        val_w = None if w is None else w[val_indices]

        # assign each unique input training value the same training label
        if len(train_X.shape) == 1:  # scalars
            u = np.unique(train_X)
            new_train_y = np.zeros_like(train_y)
            for value in u:
                mask = (train_X == value)
                new_train_y[mask] = np.mean(train_y[mask])
            train_y = new_train_y

        nearest_indices = np.argsort(dists_cv[np.ix_(val_indices, train_indices)], axis=1)

        for i, k in enumerate(ks):
            if len(train_indices) < k:
                break

            val_preds = np.mean(train_y[nearest_indices[:, :k]], axis=1)
            preds[i, val_indices] = val_preds
            group_mses[i, g] = np.average((val_preds - val_y) ** 2, weights=val_w)

    mses = np.average((preds - y) ** 2, axis=1, weights=w)  # shape [K]

    if plot:
        h = max(3, len(group_names) * 0.2)
        fig, ax = plt.subplots(1, 1, figsize=(h*2, h), constrained_layout=True)
        for g, group_label in enumerate(leftout_group_labels):
            ax.scatter(x=ks, y=group_mses[:, g], label=group_label,
                       c=[cm.tab20.colors[g % 20]])
        ax.plot(ks, mses, 'g-', label='Overall val mse')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Left-out Group')
        ax.set(xlabel='k', ylabel='mse')
        ax.set_xscale('log')
        ax.grid(True)
        if isinstance(plot, str):
            print(f'Saved KNN plot to {plot}')
            plt.savefig(plot)
        else:
            plt.show()

    best_k = ks[np.argmin(mses)]

    # assign each unique input training value the same training label
    if len(X.shape) == 1:  # scalars
        u = np.unique(X)
        new_y = np.zeros_like(y)
        for value in u:
            mask = (X == value)
            new_y[mask] = np.mean(y[mask])
        y = new_y

    nearest_indices = np.argsort(dists[np.ix_(test_indices, cv_indices)], axis=1)
    test_preds = np.mean(y[nearest_indices[:, :best_k]], axis=1)

    best_val_mse = np.min(mses)
    test_y = labels[test_indices]
    test_w = None if weights is None else weights[test_indices]
    test_mse = np.average((test_preds - test_y) ** 2, weights=test_w)
    print(f'best val mse: {best_val_mse:.3f}, best k: {best_k}, test mse: {test_mse:.3f}')
    return test_preds