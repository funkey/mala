import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t

cdef extern from "emst.h":
    void mlpack_emst(
        int numPoints,
        int numDims,
        const double* points,
        double* mst);

cdef extern from "um_loss.h":
    double c_um_loss_gradient(
        size_t numNodes,
        const double* mst,
        const int64_t* gtSeg,
        double alpha,
        double* gradients,
        double* ratioPos,
        double* ratioNeg,
        double& totalNumPairsPos,
        double& totalNumPairsNeg);

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

def um_loss(
    np.ndarray[double, ndim=2] mst,
    np.ndarray[int64_t, ndim=1] gt_seg,
    double alpha):

    cdef size_t num_points = gt_seg.shape[0]
    cdef size_t num_edges = mst.shape[0]

    assert num_points == num_edges + 1, ("Number of edges in MST is unequal "
                                         "number of points in segmentation "
                                         "minus one.")

    assert mst.shape[1] == 3, "mst not given as rows of [u, v, dist]"

    # the C++ part assumes contiguous memory, make sure we have it (and do 
    # nothing, if we do)
    if not mst.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous mst arrray (avoid this by "
              "passing C_CONTIGUOUS arrays)")
        mst = np.ascontiguousarray(mst)
    if not gt_seg.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous gt_seg arrray (avoid this by "
              "passing C_CONTIGUOUS arrays)")
        gt_seg = np.ascontiguousarray(gt_seg)

    # prepare output arrays
    cdef np.ndarray[double, ndim=1] gradients = np.zeros(
            (num_edges,),
            dtype=np.float64)
    cdef np.ndarray[double, ndim=1] ratio_neg = np.zeros(
            (num_edges,),
            dtype=np.float64)
    cdef np.ndarray[double, ndim=1] ratio_pos = np.zeros(
            (num_edges,),
            dtype=np.float64)

    cdef double num_pairs_pos;
    cdef double num_pairs_neg;

    cdef double loss = c_um_loss_gradient(
        num_points,
        &mst[0, 0],
        &gt_seg[0],
        alpha,
        &gradients[0],
        &ratio_pos[0],
        &ratio_neg[0],
        num_pairs_pos,
        num_pairs_neg)

    return (
        loss,
        gradients,
        ratio_pos,
        ratio_neg,
        num_pairs_pos,
        num_pairs_neg)
