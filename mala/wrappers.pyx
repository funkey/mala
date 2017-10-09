import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t

cdef extern from "emst.h":
    void mlpack_emst(
        int numPoints,
        int numDims,
        const double* points,
        double* mst);

cdef extern from "um_scores.h":
    void c_um_scores(
        int numNodes,
        const double* emst,
        const float* dist,
        const float* distSquared,
        const int64_t* gtSeg,
        float alpha,
        float* numPairsPos,
        float* numPairsNeg,
        float* scoresA,
        float* scoresB,
        float* scoresC);

def emst(np.ndarray[double, ndim=2] points):

    cdef int num_points = points.shape[0]
    cdef int num_dims = points.shape[1]

    # the C++ part assumes contiguous memory, make sure we have it (and do 
    # nothing, if we do)
    if not points.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous points arrray (avoid this by "
              "passing C_CONTIGUOUS arrays)")
        points = np.ascontiguousarray(points)

    # prepare emst array
    cdef np.ndarray[double, ndim=2] output = np.zeros(
            (num_points - 1, 3),
            dtype=np.float64)

    mlpack_emst(num_points, num_dims, &points[0, 0], &output[0, 0])

    return output

def um_scores(
    np.ndarray[double, ndim=2] emst,
    np.ndarray[float, ndim=1] dist,
    np.ndarray[float, ndim=1] dist_squared,
    np.ndarray[int64_t, ndim=1] gt_seg,
    float alpha):

    cdef int num_points = gt_seg.shape[0]
    cdef int num_edges = emst.shape[0]

    assert num_points == num_edges + 1, ("Number of edges in MST is unequal "
                                         "number of points in segmentation "
                                         "minus one.")

    assert emst.shape[1] == 3, "emst not given as rows of [u, v, dist]"

    # the C++ part assumes contiguous memory, make sure we have it (and do 
    # nothing, if we do)
    if not emst.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous emst arrray (avoid this by "
              "passing C_CONTIGUOUS arrays)")
        emst = np.ascontiguousarray(emst)
    if not dist.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous dist arrray (avoid this by "
              "passing C_CONTIGUOUS arrays)")
        dist = np.ascontiguousarray(dist)
    if not dist_squared.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous dist_squared arrray (avoid this by "
              "passing C_CONTIGUOUS arrays)")
        dist_squared = np.ascontiguousarray(dist_squared)
    if not gt_seg.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous gt_seg arrray (avoid this by "
              "passing C_CONTIGUOUS arrays)")
        gt_seg = np.ascontiguousarray(gt_seg)

    # prepare output arrays
    cdef np.ndarray[float, ndim=1] num_pairs_pos = np.zeros(
            (num_edges,),
            dtype=np.float32)
    cdef np.ndarray[float, ndim=1] num_pairs_neg = np.zeros(
            (num_edges,),
            dtype=np.float32)
    cdef np.ndarray[float, ndim=1] scores_a = np.zeros(
            (num_edges,),
            dtype=np.float32)
    cdef np.ndarray[float, ndim=1] scores_b = np.zeros(
            (num_edges,),
            dtype=np.float32)
    cdef np.ndarray[float, ndim=1] scores_c = np.zeros(
            (num_edges,),
            dtype=np.float32)

    c_um_scores(
        num_points,
        &emst[0, 0],
        &dist[0],
        &dist_squared[0],
        &gt_seg[0],
        alpha,
        &num_pairs_pos[0],
        &num_pairs_neg[0],
        &scores_a[0],
        &scores_b[0],
        &scores_c[0])

    return (num_pairs_pos, num_pairs_neg, scores_a, scores_b, scores_c)
