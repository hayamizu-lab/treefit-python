# Copyright (C) 2020  Momoko Hayamizu <hayamizu@ism.ac.jp>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program.  If not, see
# <http://www.gnu.org/licenses/>.

import math
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import scipy
import sklearn.decomposition

def perturbate_poisson(counts, strength=1.0):
    return (np.random.poisson(counts) * strength).astype(counts.dtype)

def calculate_distance_matrix(expression):
    # Use 'euclidean' method for now.
    # TODO: make method parameter
    return scipy.spatial.distance.cdist(expression, expression, 'euclidean')

def calculate_mst(expression):
    distance_matrix = calculate_distance_matrix(expression)
    mst = scipy.sparse.csgraph.minimum_spanning_tree(distance_matrix)
    # Remove weights
    mst[mst > 0] = 1
    return mst + mst.T

def perturbate_knn(expression, strength=1.0):
    n_samples, n_features = expression.shape
    # TODO: Improve
    k_nearest_neighbors = round(n_samples * 0.0125)
    if k_nearest_neighbors < 2:
        k_nearest_neighbors = 2
    standard_deviation = strength / np.sqrt(k_nearest_neighbors)
    distance_matrix = calculate_distance_matrix(expression)
    perturbated_expression = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        sorted_indices = np.argsort(distance_matrix[i, :])
        nearest_neighbors = \
            expression[sorted_indices[1:(k_nearest_neighbors + 1)]]
        diffs = nearest_neighbors - expression[i]
        weights = np.random.normal(scale=standard_deviation,
                                   size=(n_features,))
        weighted_diffs = diffs * weights
        perturbated_expression[i] = expression[i] + np.sum(weighted_diffs, axis=0)
    return perturbated_expression

def calculate_low_dimension_laplacian_eigenvectors(mst, p):
    laplacian = scipy.sparse.csgraph.laplacian(mst)
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian.toarray())
    while len(eigenvalues) > 0 and math.isclose(eigenvalues[0], 0, abs_tol=1e-9):
        eigenvalues = eigenvalues[1:]
        eigenvectors = eigenvectors[:, 1:]
    low_dimension_values = eigenvalues[0:p]
    low_dimension_vectors = eigenvectors[:, 0:p]
    if len(np.unique(low_dimension_values)) != len(low_dimension_values):
        low_dimension_vectors = scipy.linalg.orth(low_dimension_vectors)
    low_dimension_vectors /= np.linalg.norm(low_dimension_vectors, axis=0)
    return low_dimension_vectors

def calculate_canonical_correlation(u, v):
    uTv = np.matmul(u.T, v)
    return scipy.linalg.svd(uTv)[1]

def calculate_grassmann_distance_max_cca(canonical_correlation):
    max_cos_theta = np.max(canonical_correlation)
    return np.sqrt(np.max([0, 1 - max_cos_theta ** 2]))

def calculate_grassmann_distance_rms_cca(canonical_correlation):
    return np.sqrt(np.average(1 - canonical_correlation ** 2))

def calculate_eigenvectors_list(original,
                                perturbations,
                                normalize,
                                reduce_dimension,
                                build_tree,
                                max_p,
                                n_perturbations):
    poisson_strength = 1.0
    # TODO: Improve
    knn_strength = 0.2 * (500 / 200) ** 0.5
    targets = []

    if perturbations is None or 'poisson' in perturbations:
        counts = original.get('counts')
        if counts:
            targets.append(counts)
            for i in range(n_perturbations):
                perturbated_counts = perturbate_poisson(counts, poisson_strength)
                targets.append(perturbated_counts)
        elif not perturbations is None:
            raise TypeError('no count data: %s' % original)

        if len(targets) > 0:
            # TODO: Normalize
            pass

    if perturbations is None or 'knn' in perturbations:
        if len(targets) == 0:
            expression = original.get('expression')
            if expression is None:
                raise TypeError('no expression data: %s' % original)

            targets.append(expression)
            for i in range(n_perturbations):
                perturbated_expression = perturbate_knn(expression, knn_strength)
                targets.append(perturbated_expression)
        else:
            targets = \
                [perturbate_knn(expression, knn_strength) for target in targets]

    def calculate(target):
        if reduce_dimension is None or isinstance(reduce_dimension, int):
            if isinstance(reduce_dimension, int):
                n_dimensions = reduce_dimension
            else:
                n_dimensions = None
            pca = sklearn.decomposition.PCA(n_dimensions)
            target = pca.fit_transform(target)
        elif reduce_dimension:
            target = reduce_dimension(target)
        tree = calculate_mst(target)
        return calculate_low_dimension_laplacian_eigenvectors(tree, max_p)

    return list(map(calculate, targets))

class Fit:
    def __init__(self,
                 name,
                 max_cca_distance,
                 rms_cca_distance,
                 n_principal_paths_candidates):
        self.name = name
        self.max_cca_distance = max_cca_distance
        self.rms_cca_distance = rms_cca_distance
        self.n_principal_paths_candidates = n_principal_paths_candidates

def treefit(target,
            name=None,
            perturbations=None,
            normalize=None,
            reduce_dimension=None,
            build_tree=None,
            max_p=20,
            verbose=False,
            n_perturbations=20):
    if name is None:
        name = "fit"
    eigenvectors_list = calculate_eigenvectors_list(target,
                                                    perturbations,
                                                    normalize,
                                                    reduce_dimension,
                                                    build_tree,
                                                    max_p,
                                                    n_perturbations)
    ps = []
    max_cca_distance_means = []
    max_cca_distance_standard_deviations = []
    rms_cca_distance_means = []
    rms_cca_distance_standard_deviations = []
    for p in range(1, max_p + 1):
        ps.append(p)
        max_cca_distance_values = []
        rms_cca_distance_values = []
        for i in range(1, len(eigenvectors_list)):
            u = eigenvectors_list[0][:, 0:p]
            v = eigenvectors_list[i][:, 0:p]
            canonical_correlation = calculate_canonical_correlation(u, v)
            max_cca_distance_values.append(
                calculate_grassmann_distance_max_cca(canonical_correlation))
            rms_cca_distance_values.append(
                calculate_grassmann_distance_rms_cca(canonical_correlation))
        max_cca_distance_means.append(np.mean(max_cca_distance_values))
        max_cca_distance_standard_deviations.append(
            np.std(max_cca_distance_values))
        rms_cca_distance_means.append(np.mean(rms_cca_distance_values))
        rms_cca_distance_standard_deviations.append(
            np.std(rms_cca_distance_values))

    n_principal_paths_candidates = []
    for p in range(1, max_p - 1):
        if p == 1:
            rms_cca_distance_mean_before = float("inf")
        else:
            rms_cca_distance_mean_before = rms_cca_distance_means[p - 2]
        rms_cca_distance_mean = rms_cca_distance_means[p - 1]
        rms_cca_distance_mean_after = rms_cca_distance_means[p]
        if rms_cca_distance_mean_before > rms_cca_distance_mean and \
           rms_cca_distance_mean < rms_cca_distance_mean_after:
            n_principal_paths_candidates.append(p + 1)

    max_cca_distance = pd.DataFrame({
        'p': ps,
        'mean': max_cca_distance_means,
        'standard_deviation': max_cca_distance_standard_deviations,
    })
    rms_cca_distance = pd.DataFrame({
        'p': ps,
        'mean': rms_cca_distance_means,
        'standard_deviation': rms_cca_distance_standard_deviations,
    })
    return Fit(name,
               max_cca_distance,
               rms_cca_distance,
               n_principal_paths_candidates)

def plot(fit, *args):
    fig, axes = plt.subplots(1, 2)
    max_ax = axes[0]
    rms_ax = axes[1]

    def plot_data_frame(ax, title, value_label, data_frame):
        p = data_frame['p']
        mean = data_frame['mean']
        standard_deviation = data_frame['standard_deviation']
        ax.set_title(title)
        ax.set_xlabel('p: Dimensionality of the feature space of trees')
        ax.set_ylabel('%s (mean and SD)' % value_label)
        ax.plot(p, mean)
        ax.fill_between(p,
                        mean - standard_deviation,
                        mean + standard_deviation,
                        alpha=0.2,
                        zorder=-10)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))

    plot_data_frame(max_ax,
                    'Analysis of the structural instability\n' +
                    'of the estimated trees',
                    'max_cca_distance',
                    fit.max_cca_distance)
    plot_data_frame(rms_ax,
                    'Prediction for\nthe number of principal paths',
                    'rms_cca_distance',
                    fit.rms_cca_distance)
