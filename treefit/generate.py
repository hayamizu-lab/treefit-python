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
