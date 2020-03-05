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

import numpy as np
import pytest

import treefit
import treefit.fit

def test_caluculate_distance_matrix():
    values = np.array([[1, 0],
                       [0, 1],
                       [0, 0]])
    actual_distance_matrix = treefit.fit.calculate_distance_matrix(values)
    expected_distance_matrix = np.array([[0,          np.sqrt(2), 1],
                                         [np.sqrt(2), 0,          1],
                                         [1,          1,          0]])
    assert np.all(actual_distance_matrix == expected_distance_matrix)

def test_perturbate_knn():
    expression = treefit.data.generate_2d_n_arms_star_data(100, 3, 0.1)
    n_perturbations = 5
    strength = 0.2
    min_diff_mean = 0.001
    max_diff_mean = 0.02
    min_diff_variance = 0.00001
    max_diff_variance = 0.001
    perturbated_expression_list = \
        map(lambda i: treefit.fit.perturbate_knn(expression, strength),
            range(n_perturbations))
    original_distance_matrix = \
        treefit.fit.calculate_distance_matrix(expression)
    perturbated_distance_matrix_list = \
        map(treefit.fit.calculate_distance_matrix,
            perturbated_expression_list)
    diffs = \
        map(lambda perturbated_distance_matrix: \
                abs(original_distance_matrix - perturbated_distance_matrix),
            perturbated_distance_matrix_list)
    diffs = list(diffs)
    diffs = np.array(diffs)
    diff_means = np.mean(diffs, axis=(1, 2))
    diff_variances = np.var(diffs, axis=(1, 2))
    assert np.all(diff_means > min_diff_mean)
    assert np.all(diff_means < max_diff_mean)
    assert np.all(diff_variances > min_diff_variance)
    assert np.all(diff_variances < max_diff_variance)

def test_calculate_mst():
    values = np.array([[1, 0],
                       [0, 3],
                       [0, 0]])
    mst = treefit.fit.calculate_mst(values)
    assert mst.toarray() == \
        pytest.approx(np.array([[0,     0, 1.0],
                                [0,     0, 1.0],
                                [1.0, 1.0,   0]]))

def test_calculate_low_dimension_laplacian_eigenvectors():
    values = np.stack([range(1, 11), range(1, 11)],
                      axis=1)
    mst = treefit.fit.calculate_mst(values)
    eigenvectors = \
        treefit.fit.calculate_low_dimension_laplacian_eigenvectors(mst, 4)
    assert eigenvectors[0, :] == \
        pytest.approx(np.array([-0.44170765,
                                0.4253254,
                                -0.39847023,
                                -0.3618034]))

def test_calculate_canonical_correlation():
    values1 = np.stack([range(1, 11), range(1, 11)],
                       axis=1)
    mst1 = treefit.fit.calculate_mst(values1)
    eigenvectors1 = \
        treefit.fit.calculate_low_dimension_laplacian_eigenvectors(mst1, 4)
    values2 = np.array([[5, 1, 2, 5, 8, 9, 2, 8, 1, 2],
                        [9, 7, 4, 2, 8, 3, 1, 9, 4, 1]]).T
    mst2 = treefit.fit.calculate_mst(values2)
    eigenvectors2 = \
        treefit.fit.calculate_low_dimension_laplacian_eigenvectors(mst2, 4)
    canonical_correlation = \
        treefit.fit.calculate_canonical_correlation(eigenvectors1,
                                                    eigenvectors2)
    assert canonical_correlation == \
        pytest.approx(np.array([0.94948070,
                                0.76344509,
                                0.57019530,
                                0.06708014]))

def test_calculate_grassmann_distance_max_cca():
    canonical_correlation = np.array([0.94948070,
                                      0.76344509,
                                      0.57019530,
                                      0.06708014])
    distance = \
        treefit.fit.calculate_grassmann_distance_max_cca(canonical_correlation)
    assert distance == pytest.approx(0.3138254297)

def test_calculate_grassmann_distance_rms_cca():
    canonical_correlation = np.array([0.94948070,
                                      0.76344509,
                                      0.57019530,
                                      0.06708014])
    distance = \
        treefit.fit.calculate_grassmann_distance_rms_cca(canonical_correlation)
    assert distance == pytest.approx(0.7392590158)

def test_treefit_2_arms():
    star = treefit.data.generate_2d_n_arms_star_data(200, 2, 0.1)
    fit = treefit.treefit({"expression": star},
                          "tree-like")
    assert [fit.name, fit.n_principal_paths_candidates[0]] == ["tree-like", 2]

def test_treefit_3_arms():
    star = treefit.data.generate_2d_n_arms_star_data(200, 3, 0.1)
    fit = treefit.treefit({"expression": star},
                          "tree-like")
    assert [fit.name, fit.n_principal_paths_candidates[0]] == ["tree-like", 3]
