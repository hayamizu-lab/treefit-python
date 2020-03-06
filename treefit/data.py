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

def generate_2d_n_arms_star_data(n_samples, n_arms, fatness):
    """Generate a 2-dimensional star tree data that contain ``n_samples``
    data points and fit a star tree with ``n_arms`` arms.

    Parameters
    ----------
    n_samples : int
        The number of samples to be generated.

    n_arms : int
        The number of arms to be generated.

    fatness : float
        How fat from the based star tree. ``[0.0, 1.0]`` is available
        value range.

    Returns
    -------
    star : numpy.array

        A generated ``numpy.array``. The rows and columns correspond
        to samples and features.

    Examples
    --------
    >>> import treefit
    >>> from matplotlib.pyplot as plt
    # Generate a 2-dimensional star tree data that contain 500 data points
    # and fit a star tree with 3 arms. The generated data are a bit noisy but
    # tree-like.
    >>> star_tree_like = treefit.data.generate_2d_n_arms_star_data(500, 3, 0.1)
    >>> plt.figure()
    >>> plt.scatter(star_tree_like[:, 0], star_tree_like[:, 1])
    # Generate a 2-dimensional star tree data that contain 600 data points
    # and fit a star tree with 5 arms. The generated data are very noisy and
    # less tree-like.
    >>> star_less_tree_like = treefit.data.generate_2d_n_arms_star_data(600, 5, 0.9)
    >>> plt.figure()
    >>> plt.scatter(star_less_tree_like[:, 0], \
    ...             star_less_tree_like[:, 1])
    """
    n_features = 2
    standard_deviation = fatness / n_arms
    star = np.zeros((n_samples, n_features), np.float)
    for i in range(n_samples):
        arm = np.random.choice(range(n_arms))
        theta = (arm + 1) / n_arms * n_features * np.pi
        position = np.array([np.cos(theta), np.sin(theta)])
        position = position * np.random.uniform()
        position = position + np.random.normal(scale=standard_deviation,
                                               size=n_features)
        star[i, :] = position
    return star

def generate_2d_n_arms_linked_star_data(n_samples_list,
                                        n_arms_list,
                                        fatness):
    """Generate a 2-dimensional linked star tree data.

    Each star tree data contain ``n_samples_vector[i]`` data points and
    fit a star tree with ``n_arms_vector[i]`` arms.

    Parameters
    ----------
    n_samples_list : [int]
        The list of the number of samples to be generated. For
        example, ``[200, 100, 300]`` means that the first tree has 200
        samples, the second tree has 100 samples and the third tree
        has 300 samples.

    n_arms_list : [int]
        The list of the number of arms to be generated. For example,
        ``[3, 2, 5]`` means the first tree fits a star tree with 3
        arms, the second tree fits a star tree with 2 arms and the
        third tree fits a star tree with 5 arms. The length of
        ``n_arms_list`` must equal to the length of
        ``n_samples_list``.

    fatness : [float]
        How fat from the based tree. ``[0.0, 1.0]`` is available value
        range.

    Returns
    -------
    linked_star : numpy.array

        A generated `numpy.array`. The rows and columns correspond to
        samples and features.

    Examples
    --------
    >>> import treefit
    >>> from matplotlib.pyplot as plt
    # Generate a 2-dimensional linked star tree data that contain
    # 200-400-300 data points and fit a linked star tree with 3-5-4
    # arms. The generated data are a bit noisy but tree-like.
    >>> linked_star_tree_like = \
    ...     treefit.data.generate_2d_n_arms_linked_star_data([200, 400, 300],
    ...                                                      [3, 5, 4],
    ...                                                      0.1)
    >>> plt.figure()
    >>> plt.scatter(linked_star_tree_like[:, 0],
    ...             linked_star_tree_like[:, 1])
    # Generate a 2-dimensional linked star tree data that contain
    # 300-200 data points and fit a linked star tree with 4-3 arms.
    # The generated data are very noisy and less tree-like.
    >>> linked_star_less_tree_like = \
    ...     treefit.data.generate_2d_n_arms_linked_star_data([300, 200],
    ...                                                      [4, 3],
    ...                                                      0.9)
    >>> plt.figure()
    >>> plt.scatter(linked_star_less_tree_like[:, 0],
    ...             linked_star_less_tree_like[:, 1])
    """

    n_features = 2
    n_total_samples = np.sum(n_samples_list)
    star = np.zeros((n_total_samples, n_features), np.float)
    n_samples_offset = 0
    sub_star_offsets = [0.0, 0.0]
    for i in range(len(n_samples_list)):
        n_samples = n_samples_list[i]
        n_arms = n_arms_list[i]
        sub_star = generate_2d_n_arms_star_data(n_samples, n_arms, fatness)
        theta = 2 * np.pi * (n_arms // 2 / n_arms)
        sub_star_offsets[0] = sub_star_offsets[0] + -np.cos(theta) + 1
        sub_star_offsets[1] = sub_star_offsets[1] + -np.sin(theta)
        sub_star[:, 0] = sub_star[:, 0] + sub_star_offsets[0]
        sub_star[:, 1] = sub_star[:, 1] + sub_star_offsets[1]
        star[n_samples_offset:(n_samples_offset+n_samples), :] = sub_star
        n_samples_offset += n_samples
    return star
