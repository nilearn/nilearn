"""
Comparing different connectivity measures
=========================================

This example shows how to extract signals from regions defined by an atlas,
and to estimate different connectivity measures based on these signals.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Copied from matplotlib 1.2.0 for matplotlib 0.99 compatibility.
_bwr_data = ((0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0))
plt.cm.register_cmap(cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
    "bwr", _bwr_data))


def corr_to_Z(corr):
    """Applies Z-Fisher transform. """
    Z = corr.copy()  # avoid side effects
    corr_is_one = 1.0 - abs(corr) < 1e-15
    Z[corr_is_one] = np.inf * np.sign(Z[corr_is_one])
    Z[np.logical_not(corr_is_one)] = \
        np.arctanh(corr[np.logical_not(corr_is_one)])
    return Z


def plot_matrix(mean_conn, title="connectivity", ticks=[], tick_labels=[]):
    """Plot connectivity matrix, for a given measure. """

    mean_conn = mean_conn.copy()

    # Put zeros on the diagonal, for graph clarity
    size = mean_conn.shape[0]
    mean_conn[range(size), range(size)] = 0
    vmax = np.abs(mean_conn).max()
    if vmax <= 2e-16:
        vmax = 0.1

    # Display connectivity matrix
    plt.figure()
    plt.imshow(mean_conn, interpolation="nearest",
               vmin=-vmax, vmax=vmax, cmap=plt.cm.get_cmap("bwr"))
    plt.colorbar()
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    plt.xticks(ticks, tick_labels, size=8, rotation=90)
    plt.yticks(ticks, tick_labels, size=8)
    ax.yaxis.tick_left()

    plt.title(title)


def scatterplot_matrix(coefs1, coefs2, coefs_ref, names,
                       title1='measure 1', title2='measure 2',
                       title_ref='reference measure'):
    """Plot a scatterplot matrix of subplots. Each connectivity coefficient is
    scatter plotted for two given measures against a reference measure. The
    line of best fit is plotted for significantly correlated coefficients. """
    n_subjects = coefs1.shape[0]
    n_coefs = coefs1[0].size
    fig, axes = plt.subplots(nrows=n_coefs, ncols=n_coefs, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')

    # Plot the coefs
    from scipy.stats.stats import pearsonr
    all_coefs = [coefs1, coefs2, coefs_ref, coefs_ref]
    colors = ['b', 'g', 'r', 'r']
    plots = []
    for n, coefs in enumerate(all_coefs):
        coefs = coefs.reshape(n_subjects, -1)
        for x, y in zip(*np.triu_indices_from(axes, k=1)):
            indices = [(x, y), (y, x), (x, y), (y, x)]
            id_x, id_y = indices[n]
            plot = axes[id_x, id_y].scatter(coefs[:, id_x], coefs[:, id_y],
                                            c=colors[n])
            # Plot line of best fit if significative Pearson correlation
            rho, p_val = pearsonr(coefs[:, id_x], coefs[:, id_y])
            if p_val < 0.05 / coefs.shape[-1]:
                fit = np.polyfit(coefs[:, id_x], coefs[:, id_y], 1)
                fit_fn = np.poly1d(fit)
                axes[id_x, id_y].plot(coefs[:, id_x], fit_fn(coefs[:, id_x]),
                                      linestyle='-', c=colors[n])
        plots.append(plot)

    plt.figlegend(plots, [title1, title2, title_ref],
                  loc='lower center', mode='expand', ncol=3, borderaxespad=0.2)

    # Label the diagonal subplots
    for i, label in enumerate(names):
        axes[i, i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                            ha='center', va='center')

    # Turn on the proper x or y axes ticks
    import itertools
    for i, j in zip(range(n_coefs), itertools.cycle((-1, 0))):
        axes[j, i].xaxis.set_visible(True)
        axes[i, j].yaxis.set_visible(True)
        xlabels = axes[j, i].get_xticklabels()
        plt.setp(xlabels, rotation=90)
    plt.tight_layout(pad=4.5, w_pad=0., h_pad=0.)
    fig.suptitle("connectivity scatter plots")

print("-- Fetching datasets ...")
import nilearn.datasets
atlas = nilearn.datasets.fetch_msdl_atlas()
dataset = nilearn.datasets.fetch_adhd()

import nilearn.image
import nilearn.input_data

import joblib
mem = joblib.Memory("/home/sb238920/CODE/Parietal/nilearn/nilearn_cache/adhd")

# Number of subjects to consider for connectivity computations
n_subjects = 20
subjects = []
for subject_n in range(n_subjects):
    filename = dataset["func"][subject_n]
    print("Processing file %s" % filename)

    print("-- Computing confounds ...")
    confound_file = dataset["confounds"][subject_n]
    hv_confounds = mem.cache(nilearn.image.high_variance_confounds)(filename)

    print("-- Computing region signals ...")
    masker = nilearn.input_data.NiftiMapsMasker(
        atlas["maps"], resampling_target="maps", detrend=True,
        low_pass=None, high_pass=0.01, t_r=2.5, standardize=False,
        memory=mem, memory_level=1, verbose=1)
    region_ts = masker.fit_transform(filename,
                                     confounds=[hv_confounds, confound_file])
    subjects.append(region_ts)


import nilearn.connectivity
print("-- Measuring connecivity ...")
all_matrices = []
mean_matrices = []
for kind in ['correlation', 'partial correlation', 'tangent']:
    estimator = {'kind': kind}
    cov_embedding = nilearn.connectivity.CovEmbedding(**estimator)
    matrices = nilearn.connectivity.vec_to_sym(
        cov_embedding.fit_transform(subjects))
    all_matrices.append(matrices)
    if kind == 'tangent':
        mean = nilearn.connectivity.embedding.cov_to_corr(
            cov_embedding.mean_cov_)
    else:
        mean = matrices.mean(axis=0)
    mean_matrices.append(mean)

print("-- Displaying results")
regions = ['L DMN', 'med DMN', 'front DMN', 'R DMN']
titles = ['correlations mean', 'partial correlations mean',
          'geometric mean']
for matrix, title in zip(mean_matrices, titles):
    plot_matrix(matrix, title=title, ticks=range(3, 7), tick_labels=regions)

Z_correlations = corr_to_Z(all_matrices[0])
Z_partials = corr_to_Z(all_matrices[1])
tangents = all_matrices[2]

# Scatter plot connectivity coefficients between some regions of the Default
# Mode Network
scatterplot_matrix(Z_correlations[:, 3:5, 5:7],
                   Z_partials[:, 3:5, 5:7],
                   tangents[:, 3:5, 5:7],
                   names=['L DMN/\nfront DMN', 'L DMN/\nR DMN',
                          'med DMN/\nfront DMN', 'med DMN/\nR DMN'],
                   title1='correlation\n(Z-transformed)',
                   title2='partial correlation\n(Z- transformed)',
                   title_ref='tangent')
plt.show()