import tensorflow as tf
import numpy as np
import mala
from .py_func_gradient import py_func_gradient

def get_emst(embedding):

    return mala.emst(embedding.astype(np.float64))

def get_emst_op(embedding, name=None):

    return tf.py_func(
        get_emst,
        [embedding],
        [tf.float64],
        name=name)[0]

def get_um_loss(mst, gt_seg, alpha):
    '''Compute the ultra-metric loss and gradient given an MST and
    segmentation.

    Args:

        mst (Tensor, shape ``(3, n-1)``): u, v indices and distance of edges of
            the MST spanning n nodes.

        gt_seg (Tensor, arbitrary shape): The label of each node. Will be
            flattened. The indices in mst should be valid indices into this
            array.

    Returns:

        A tuple::

            (loss, gradients, num_pairs_pos, num_pairs_neg)

        Except for ``loss``, each entry is a tensor of shape ``(n-1,)``,
        corresponding to the edges in the MST. ``gradients`` contains the
        gradients on the edge distances. ``num_pairs_pos`` and
        ``num_pairs_neg`` the number of positive and negative pairs that share
        an edge.
    '''

    return mala.um_loss(
        mst,
        gt_seg.flatten(),
        alpha)

class UmLoss:
    '''Wrapper class to avoid re-computation of the UM loss between forward and
    backward passes. This class will store the results of the forward pass, and
    reuse it in the backward pass.
    '''

    def __init__(self):
        self.__loss = None
        self.__gradient = None

    def loss(self, mst, dist, gt_seg, alpha):

        # We don't use 'dist' here, it is already contained in the mst. It is
        # passed here just so that tensorflow knows there is dependecy to the
        # ouput.

        print("Calling UmLoss::loss")
        (
            self.__loss,
            self.__gradient,
            self.__num_pairs_pos,
            self.__num_pairs_neg) = get_um_loss(mst, gt_seg, alpha)

        # ensure that every float tf sees is float32
        self.__loss = np.float32(self.__loss)
        self.__gradient = self.__gradient.astype(np.float32)

        return (
            self.__loss,
            self.__num_pairs_pos,
            self.__num_pairs_neg)

    def gradient(self, op, dloss, dpairs_pos, dpairs_neg):

        print("Calling UmLoss::gradient")
        # the loss has only a gradient wrt. the distance
        return (None, tf.convert_to_tensor(self.__gradient)*dloss, None, None)

def ultrametric_loss_op(
        embedding,
        gt_seg,
        alpha=0.1,
        add_coordinates=True,
        name=None):
    '''Returns a tensorflow op to compute the ultra-metric quadrupel loss::

        L = sum_p sum_n max(0, d(n) - d(p) + alpha)^2

    where ``p`` and ``n`` are pairs points with same and different labels,
    respectively, and ``d(.)`` the distance between the points.

    Args:

        embedding (Tensor, shape ``(k, d, h, w)``): A k-dimensional feature
            embedding of points in 3D.

        gt_seg (Tensor, shape ``(d, h, w)``): The ground-truth labels of the
            points.

        alpha (float): The margin term of the quadrupel loss.

        add_coordinates(bool): If ``True``, add the ``(z, y, x)`` coordinates
            of the points to the embedding.

        name (string): An optional name for the operator.
    '''

    alpha = tf.constant(alpha)

    # We get the embedding as a tensor of shape (k, d, h, w).
    depth, height, width = embedding.shape.as_list()[-3:]

    # 1. Augmented by spatial coordinates, if requested.

    if add_coordinates:
        coordinates = tf.meshgrid(
            range(depth),
            range(height),
            range(width),
            indexing='ij')
        for i in range(len(coordinates)):
            coordinates[i] = tf.cast(coordinates[i], tf.float32)
        embedding = tf.concat([embedding, coordinates], 0)

    # 2. Transpose into tensor (d*h*w, k+3), i.e., one embedding vector per
    #    node, augmented by spatial coordinates if requested.

    embedding = tf.transpose(embedding, perm=[1, 2, 3, 0])
    embedding = tf.reshape(embedding, [depth*width*height, -1])

    # 3. Get the EMST on the embedding vectors.

    emst = get_emst_op(embedding)

    # 4. Compute the lengths of EMST edges

    edges_u = tf.gather(embedding, tf.cast(emst[:,0], tf.int64))
    edges_v = tf.gather(embedding, tf.cast(emst[:,1], tf.int64))
    dist_squared = tf.reduce_sum(tf.square(tf.subtract(edges_u, edges_v)), 1)
    dist = tf.sqrt(dist_squared)

    # 5. Compute the UM loss

    um_loss = UmLoss()
    loss = py_func_gradient(
        lambda m, d, g, a, l=um_loss: l.loss(m, d, g, a),
        [emst, dist, gt_seg, alpha],
        [tf.float32, tf.int64, tf.int64],
        gradient=lambda o, lo, p, n, l=um_loss: l.gradient(o, lo, p, n),
        name=name)[0]

    return (loss, emst, edges_u, edges_v, dist)
