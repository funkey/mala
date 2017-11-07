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
        name=name,
        stateful=False)[0]

def get_um_loss(mst, dist, gt_seg, alpha):
    '''Compute the ultra-metric loss given an MST and segmentation.

    Args:

        mst (Tensor, shape ``(3, n-1)``): u, v indices and distance of edges of
            the MST spanning n nodes.

        dist (Tensor, shape ``(n-1)``): The distances of the edges. This
            argument will be ignored, it is used only to communicate to
            tensorflow that there is a dependency on distances. The distances
            actually used are the ones in parameter ``mst``.

        gt_seg (Tensor, arbitrary shape): The label of each node. Will be
            flattened. The indices in mst should be valid indices into this
            array.

        alpha (Tensor, single float): The margin value of the quadrupel loss.

    Returns:

        A tuple::

            (loss, ratio_pos, ratio_neg)

        Except for ``loss``, each entry is a tensor of shape ``(n-1,)``,
        corresponding to the edges in the MST. ``ratio_pos`` and ``ratio_neg``
        are the ratio of positive and negative pairs that share an edge, of the
        total number of positive and negative pairs.
    '''

    # We don't use 'dist' here, it is already contained in the mst. It is
    # passed here just so that tensorflow knows there is dependecy to the
    # ouput.
    (loss, _, ratio_pos, ratio_neg, num_pairs_pos, num_pairs_neg) = mala.um_loss(
        mst,
        gt_seg.flatten(),
        alpha)

    return (
        np.float32(loss),
        ratio_pos.astype(np.float32),
        ratio_neg.astype(np.float32),
        np.float32(num_pairs_pos),
        np.float32(num_pairs_neg))

def get_um_loss_gradient(mst, dist, gt_seg, alpha):
    '''Compute the ultra-metric loss gradient given an MST and segmentation.

    Args:

        mst (Tensor, shape ``(3, n-1)``): u, v indices and distance of edges of
            the MST spanning n nodes.

        dist (Tensor, shape ``(n-1)``): The distances of the edges. This
            argument will be ignored, it is used only to communicate to
            tensorflow that there is a dependency on distances. The distances
            actually used are the ones in parameter ``mst``.

        gt_seg (Tensor, arbitrary shape): The label of each node. Will be
            flattened. The indices in mst should be valid indices into this
            array.

        alpha (Tensor, single float): The margin value of the quadrupel loss.

    Returns:

        A Tensor containing the gradient on the distances.
    '''

    # We don't use 'dist' here, it is already contained in the mst. It is
    # passed here just so that tensorflow knows there is dependecy to the
    # ouput.
    (_, gradient, _, _, _, _) = mala.um_loss(
        mst,
        gt_seg.flatten(),
        alpha)

    return gradient.astype(np.float32)

def get_um_loss_gradient_op(
        op,
        dloss,
        dratio_pos,
        dratio_neg,
        dnum_pairs_pos,
        dnum_pairs_neg):

    gradient = tf.py_func(
        get_um_loss_gradient,
        [x for x in op.inputs],
        [tf.float32],
        stateful=False)[0]

    return (None, gradient*dloss, None, None)

def ultrametric_loss_op(
        embedding,
        gt_seg,
        alpha=0.1,
        add_coordinates=True,
        coordinate_scale=1.0,
        pretrain=False,
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

        coordinate_scale(float or tuple of float): How to scale the
            coordinates, if used to augment the embedding.

        name (string): An optional name for the operator.
    '''

    alpha = tf.constant(alpha, dtype=tf.float32)

    # We get the embedding as a tensor of shape (k, d, h, w).
    depth, height, width = embedding.shape.as_list()[-3:]

    # 1. Augmented by spatial coordinates, if requested.

    if add_coordinates:

        try:
            scale = tuple(coordinate_scale)
        except:
            scale = (coordinate_scale,)*3

        coordinates = tf.meshgrid(
            np.arange(0, depth*scale[0], scale[0]),
            np.arange(0, height*scale[1], scale[1]),
            np.arange(0, width*scale[2], scale[2]),
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

    if pretrain:

        # we need the um_loss just to get the ratio_pos, ratio_neg, and the
        # total number of positive and negative pairs
        _, ratio_pos, ratio_neg, num_pairs_pos, num_pairs_neg = tf.py_func(
            get_um_loss,
            [emst, dist, gt_seg, alpha],
            [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
            name=name,
            stateful=False)

        loss_pos = tf.multiply(
            dist_squared,
            ratio_pos)
        loss_neg = tf.multiply(
            tf.square(tf.maximum(0.0, alpha - dist)),
            ratio_neg)

        sum_pos = tf.reduce_sum(loss_pos)*num_pairs_pos
        sum_neg = tf.reduce_sum(loss_neg)*num_pairs_neg
        num_pairs = num_pairs_pos + num_pairs_neg

        loss = (sum_pos + sum_neg)/num_pairs

    else:

        loss = py_func_gradient(
            get_um_loss,
            [emst, dist, gt_seg, alpha],
            [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
            gradient_op=get_um_loss_gradient_op,
            name=name,
            stateful=False)[0]

    return (loss, emst, edges_u, edges_v, dist)
