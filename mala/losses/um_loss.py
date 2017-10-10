import tensorflow as tf
import numpy as np
import mala

def get_emst(embedding):

    return mala.emst(embedding.astype(np.float64))

def get_emst_op(embedding, name=None):

    return tf.py_func(
        get_emst,
        [embedding],
        [tf.float64],
        name=name)[0]

def get_um_loss(mst, gt_seg, alpha):
    '''Compute the ultra-metric scores given an MST and segmentation.

    Args:

        mst (Tensor, shape ``(2, n-1)``): u and v indices of edges of the MST
            spanning n nodes.

        gt_seg (Tensor, arbitrary shape): The label of each node. Will be
            flattened. The indices in mst should be valid indices into this
            array.

    Returns:

        A tuple::

            (num_pairs_pos, scores_a, scores_b, scores_c)

        Eeach entry is a tensor of shape ``(n-1,)``, corresponding to the edges
        in the MST.

        These are the scores that need to be multiplied with their respective
        edge coefficients, along with the number of times that a positive pair
        has an MST edge as minimax edge.

        Although derived from the distances, the numbers returned here have a
        zero-gradient wrt. the distance.
    '''

    return mala.um_loss(
        mst,
        gt_seg.flatten(),
        alpha)

def get_um_loss_op(mst, gt_seg, alpha, name=None):

    return tf.py_func(
        get_um_loss,
        [mst, gt_seg, alpha],
        [tf.float64, tf.float64, tf.int64, tf.int64],
        name=name)

def ultrametric_loss_op(
        embedding,
        gt_seg,
        alpha=0.1,
        name=None,
        add_coordinates=True):
    '''Returns a tensorflow op to compute the ultra-metric loss.'''

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

    # 4. Compute the squared lengths of EMST edges

    edges_u = tf.gather(embedding, tf.cast(emst[:,0], tf.int64))
    edges_v = tf.gather(embedding, tf.cast(emst[:,1], tf.int64))
    dist_squared = tf.reduce_sum(tf.square(tf.subtract(edges_u, edges_v)), 1)
    dist = tf.sqrt(dist_squared)

    # 5. Compute the UM loss

    loss, gradients, _, _ = get_um_loss_op(emst, gt_seg, alpha)

    return (loss, embedding, emst, edges_u, edges_v, dist_squared, dist,
            gradients)
