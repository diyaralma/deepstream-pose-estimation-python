# Copyright 2020 - NVIDIA Corporation
# SPDX-License-Identifier: MIT

import numpy as np
from typing import List, Tuple
import queue as queue_module
import math

# Type aliases to match C++ template definitions
# Vec1D<T> equivalent
Vec1DInt = List[int]
Vec1DFloat = List[float]

# Vec2D<T> equivalent
Vec2DInt = List[List[int]]
Vec2DFloat = List[List[float]]

# Vec3D<T> equivalent
Vec3DInt = List[List[List[int]]]
Vec3DFloat = List[List[List[float]]]

# Constants
EPS = 1e-6
M = 2

# Topology definition
topology = [
    [0, 1, 15, 13],
    [2, 3, 13, 11],
    [4, 5, 16, 14],
    [6, 7, 14, 12],
    [8, 9, 11, 12],
    [10, 11, 5, 7],
    [12, 13, 6, 8],
    [14, 15, 7, 9],
    [16, 17, 8, 10],
    [18, 19, 1, 2],
    [20, 21, 0, 1],
    [22, 23, 0, 2],
    [24, 25, 1, 3],
    [26, 27, 2, 4],
    [28, 29, 3, 5],
    [30, 31, 4, 6],
    [32, 33, 17, 0],
    [34, 35, 17, 5],
    [36, 37, 17, 6],
    [38, 39, 17, 11],
    [40, 41, 17, 12]
]


class NvDsInferDims:
    """Python equivalent of NvDsInferDims structure"""

    def __init__(self):
        self.d = [0, 0, 0, 0]  # Dimensions array


def find_peaks(cmap_data: np.ndarray, cmap_dims: NvDsInferDims,
               threshold: float, window_size: int, max_count: int) -> Tuple[Vec1DInt, Vec3DInt]:
    """
    Method to find peaks in the output tensor. 'window_size' represents how many pixels we are considering at once to find a maximum value, or a 'peak'.
    Once we find a peak, we mark it using the 'is_peak' boolean in the inner loop and assign this maximum value to the center pixel of our window.
    This is then repeated until we cover the entire frame.

    Args:
        cmap_data: Input confidence map data as numpy array
        cmap_dims: Dimensions of the confidence map
        threshold: Threshold value for peak detection
        window_size: Size of the window for peak detection
        max_count: Maximum number of peaks to detect

    Returns:
        Tuple of (counts_out, peaks_out) where counts_out is a list of counts per channel
        and peaks_out is a 3D list of peak coordinates
    """
    w = window_size // 2
    width = cmap_dims.d[2]
    height = cmap_dims.d[1]

    # Initialize output arrays
    counts_out = [0] * cmap_dims.d[0]
    peaks_out = [[[0 for _ in range(M)] for _ in range(max_count)] for _ in range(cmap_dims.d[0])]

    for c in range(cmap_dims.d[0]):
        count = 0
        # Get the slice of data for channel c
        cmap_data_c = cmap_data.flatten()[c * width * height:(c + 1) * width * height]

        for i in range(height):
            if count >= max_count:
                break

            for j in range(width):
                if count >= max_count:
                    break

                value = cmap_data_c[i * width + j]

                if value < threshold:
                    continue

                ii_min = i - w
                jj_min = j - w
                ii_max = i + w + 1
                jj_max = j + w + 1

                if ii_min < 0:
                    ii_min = 0
                if ii_max > height:
                    ii_max = height
                if jj_min < 0:
                    jj_min = 0
                if jj_max > width:
                    jj_max = width

                is_peak = True
                for ii in range(ii_min, ii_max):
                    for jj in range(jj_min, jj_max):
                        if cmap_data_c[ii * width + jj] > value:
                            is_peak = False

                if is_peak:
                    peaks_out[c][count][0] = i
                    peaks_out[c][count][1] = j
                    count += 1

        counts_out[c] = count

    return counts_out, peaks_out


def refine_peaks(counts: Vec1DInt, peaks: Vec3DInt, cmap_data: np.ndarray,
                 cmap_dims: NvDsInferDims, window_size: int) -> Vec3DFloat:
    """
    Normalize the peaks found in 'find_peaks' and apply non-maximal suppression

    Args:
        counts: List of peak counts per channel
        peaks: 3D list of peak coordinates
        cmap_data: Input confidence map data as numpy array
        cmap_dims: Dimensions of the confidence map
        window_size: Size of the window for refinement

    Returns:
        3D list of refined peak coordinates (normalized floats)
    """
    w = window_size // 2
    width = cmap_dims.d[2]
    height = cmap_dims.d[1]

    # Initialize refined_peaks with same structure as peaks but with float values
    refined_peaks = [[[0.0 for _ in range(len(peaks[0][0]))] for _ in range(len(peaks[0]))]
                     for _ in range(len(peaks))]

    for c in range(cmap_dims.d[0]):
        count = counts[c]
        refined_peaks_a_bc = refined_peaks[c]
        peaks_a_bc = peaks[c]
        cmap_data_c = cmap_data.flatten()[c * width * height:(c + 1) * width * height]

        for p in range(count):
            refined_peak = refined_peaks_a_bc[p]
            peak = peaks_a_bc[p]

            i = peak[0]
            j = peak[1]
            weight_sum = 0.0

            for ii in range(i - w, i + w + 1):
                ii_idx = ii

                if ii < 0:
                    ii_idx = -ii
                elif ii >= height:
                    ii_idx = height - (ii - height) - 2

                for jj in range(j - w, j + w + 1):
                    jj_idx = jj

                    if jj < 0:
                        jj_idx = -jj
                    elif jj >= width:
                        jj_idx = width - (jj - width) - 2

                    weight = cmap_data_c[ii_idx * width + jj_idx]
                    refined_peak[0] += weight * ii
                    refined_peak[1] += weight * jj
                    weight_sum += weight

            refined_peak[0] /= weight_sum
            refined_peak[1] /= weight_sum
            refined_peak[0] += 0.5
            refined_peak[1] += 0.5
            refined_peak[0] /= height
            refined_peak[1] /= width

    return refined_peaks


def paf_score_graph(paf_data: np.ndarray, paf_dims: NvDsInferDims,
                    topology_param: Vec2DInt, counts: Vec1DInt,
                    peaks: Vec3DFloat, num_integral_samples: int) -> Vec3DFloat:
    """
    Create a bipartite graph to assign detected body-parts to a unique person in the frame. This method also takes care of finding the line integral to assign scores
    to these points

    Args:
        paf_data: Part Affinity Field data as numpy array
        paf_dims: Dimensions of the PAF data
        topology_param: Topology definition for body part connections
        counts: List of peak counts per channel
        peaks: 3D list of refined peak coordinates
        num_integral_samples: Number of samples for line integral calculation

    Returns:
        3D list of scores for graph connections
    """
    K = len(topology_param)
    H = paf_dims.d[1]
    W = paf_dims.d[2]
    max_count = len(peaks[0])

    # Initialize score_graph
    score_graph = [[[0.0 for _ in range(max_count)] for _ in range(max_count)] for _ in range(K)]

    for k in range(K):
        score_graph_nk = score_graph[k]
        paf_i_idx = topology_param[k][0]
        paf_j_idx = topology_param[k][1]
        cmap_a_idx = topology_param[k][2]
        cmap_b_idx = topology_param[k][3]

        # Get PAF data slices
        paf_i = paf_data.flatten()[paf_i_idx * H * W:(paf_i_idx + 1) * H * W]
        paf_j = paf_data.flatten()[paf_j_idx * H * W:(paf_j_idx + 1) * H * W]

        counts_a = counts[cmap_a_idx]
        counts_b = counts[cmap_b_idx]
        peaks_a = peaks[cmap_a_idx]
        peaks_b = peaks[cmap_b_idx]

        for a in range(counts_a):
            # Point A
            pa_i = peaks_a[a][0] * H
            pa_j = peaks_a[a][1] * W

            for b in range(counts_b):
                # Point B
                pb_i = peaks_b[b][0] * H
                pb_j = peaks_b[b][1] * W

                # Vector from Point A to Point B
                pab_i = pb_i - pa_i
                pab_j = pb_j - pa_j

                # Normalized Vector from Point A to Point B
                pab_norm = math.sqrt(pab_i * pab_i + pab_j * pab_j) + EPS
                uab_i = pab_i / pab_norm
                uab_j = pab_j / pab_norm

                integral = 0.0
                increment = 1.0 / num_integral_samples

                for t in range(num_integral_samples):
                    # Integral Point T
                    progress = float(t) / float(num_integral_samples)
                    pt_i = pa_i + progress * pab_i
                    pt_j = pa_j + progress * pab_j

                    # Convert to Integer
                    pt_i_int = int(pt_i)
                    pt_j_int = int(pt_j)

                    # Edge cases for if the point is out of bounds, just skip them
                    if pt_i_int < 0:
                        continue
                    if pt_i_int > H:
                        continue
                    if pt_j_int < 0:
                        continue
                    if pt_j_int > W:
                        continue

                    # Vector at integral point
                    pt_paf_i = paf_i[pt_i_int * W + pt_j_int]
                    pt_paf_j = paf_j[pt_i_int * W + pt_j_int]

                    # Dot Product Normalized A->B with PAF Vector
                    dot = pt_paf_i * uab_i + pt_paf_j * uab_j
                    integral += dot

                    progress += increment

                # Normalize the integral with respect to the number of samples
                integral /= num_integral_samples
                score_graph_nk[a][b] = integral

    return score_graph


def assignment(score_graph: Vec3DFloat, topology_param: Vec2DInt,
               counts: Vec1DInt, score_threshold: float, max_count: int) -> Vec3DInt:
    """
    This method takes care of solving the graph assignment problem using Munkres algorithm. Munkres algorithm is defined in 'munkres_algorithm.cpp'

    Args:
        score_graph: 3D list of scores for graph connections
        topology_param: Topology definition for body part connections
        counts: List of peak counts per channel
        score_threshold: Threshold for valid connections
        max_count: Maximum number of connections

    Returns:
        3D list of connection assignments
    """
    # Import the required modules (assumed to be translated separately)
    from pair_graph import PairGraph
    from munkres_algorithm import munkres_algorithm

    K = len(topology_param)

    # Initialize connections
    connections = [[[-1 for _ in range(max_count)] for _ in range(M)] for _ in range(K)]

    # Create cost graph by negating score graph
    cost_graph = [[[0.0 for _ in range(len(score_graph[0][0]))]
                   for _ in range(len(score_graph[0]))]
                  for _ in range(len(score_graph))]

    for i in range(len(cost_graph)):
        for j in range(len(cost_graph[i])):
            for k in range(len(cost_graph[i][j])):
                cost_graph[i][j][k] = -score_graph[i][j][k]

    cost_graph_out_a = cost_graph

    for k in range(K):
        cmap_a_idx = topology_param[k][2]
        cmap_b_idx = topology_param[k][3]
        nrows = counts[cmap_a_idx]
        ncols = counts[cmap_b_idx]
        star_graph = PairGraph(nrows, ncols)
        cost_graph_out_a_nk = cost_graph_out_a[k]
        munkres_algorithm(cost_graph_out_a_nk, star_graph, nrows, ncols)

        connections_a_nk = connections[k]
        score_graph_a_nk = score_graph[k]

        for i in range(nrows):
            for j in range(ncols):
                if star_graph.isPair(i, j) and score_graph_a_nk[i][j] > score_threshold:
                    connections_a_nk[0][i] = j
                    connections_a_nk[1][j] = i

    return connections


def connect_parts(connections: Vec3DInt, topology_param: Vec2DInt,
                  counts: Vec1DInt, max_count: int) -> Vec2DInt:
    """
    This method takes care of connecting all the body parts detected to each other
    after finding the relationships between them in the 'assignment' method

    Args:
        connections: 3D list of connection assignments
        topology_param: Topology definition for body part connections
        counts: List of peak counts per channel
        max_count: Maximum number of objects to detect

    Returns:
        2D list of connected objects with body part indices
    """
    K = len(topology_param)
    C = len(counts)

    # Initialize visited array
    visited = [[0 for _ in range(max_count)] for _ in range(C)]

    # Initialize objects array
    objects = [[-1 for _ in range(C)] for _ in range(max_count)]

    num_objects = 0
    for c in range(C):
        if num_objects >= max_count:
            break

        count = counts[c]

        for i in range(count):
            if num_objects >= max_count:
                break

            q = queue_module.Queue()
            new_object = False
            q.put((c, i))

            while not q.empty():
                node = q.get()
                c_n = node[0]
                i_n = node[1]

                if visited[c_n][i_n]:
                    continue

                visited[c_n][i_n] = 1
                new_object = True
                objects[num_objects][c_n] = i_n

                for k in range(K):
                    c_a = topology_param[k][2]
                    c_b = topology_param[k][3]

                    if c_a == c_n:
                        i_b = connections[k][0][i_n]
                        if i_b >= 0:
                            q.put((c_b, i_b))

                    if c_b == c_n:
                        i_a = connections[k][1][i_n]
                        if i_a >= 0:
                            q.put((c_a, i_a))

            if new_object:
                num_objects += 1

    # Resize objects to actual number of objects detected
    objects = objects[:num_objects]
    return objects
