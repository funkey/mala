import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

def aggregate(stats, neighborhood, separable=False):
    '''Convolve stats, such that each point contains a weighted sum of the
    stats around it.'''

    # What exactly does tf.nn.convolution do?
    #
    # inputs are:
    #   input  (1, q, d, h, w)
    #   filter (d, h, w, q, k)
    #
    # output is:
    #   output (1, k, d, h, w)
    #
    # Things are pretty clear except for the channel part. Why does the filter
    # need to have extra q and k dimensions?
    #
    #   output[1, k, z, y, x] = sum_q filter[:, :, :, q, k]*input[1, q, :, :, :]
    #
    #   For every output channel there is a set of 3D filters, one for each
    #   input channel. The output channel is the sum of the convolutions of this
    #   set.
    #   If you just wanted to apply channel-wise same convolution, you'd have to
    #   create a filter of size (d, h, w, q, q), and each filter[:, :, :, q, k]
    #   with q!=k would be zero, with q==k would be the actual filter.
    #
    #   This seems very wasteful.
    #
    # Instead, let's use the batch dimension for "channel" wise convolution.

    if not separable:

        agg = tf.nn.convolution(
            stats,
            neighborhood,
            padding='SAME',
            data_format='NCDHW')
    else:

        x = tf.nn.convolution(
            stats,
            neighborhood,
            padding='SAME',
            data_format='NCDHW')
        xy = tf.nn.convolution(
            x,
            tf.transpose(neighborhood, perm=[0, 2, 1, 3, 4]),
            padding='SAME',
            data_format='NCDHW')
        agg = tf.nn.convolution(
            xy,
            tf.transpose(neighborhood, perm=[2, 0, 1, 3, 4]),
            padding='SAME',
            data_format='NCDHW')

    return agg

def mask_loss(
    embedding,
    embedding_sum_squares,
    mask_pos,
    mask_fg,
    neighborhood,
    separable):
    '''
    Compute the mask loss for the given object only. Returns postive and negative
    contributions and the count of positive and negative pairs (weighted by
    ``neighborhood``).
    '''

    # create the negative label mask (voxels that are not in mask_pos and not
    # background)
    mask_neg = (1.0 - mask_pos)*mask_fg

    # aggregate s_0, s_1, and s_2 scores from mask_pos, embedding, and
    # embedding_sum_squares
    s_0 = aggregate(mask_pos, neighborhood, separable)
    s_1 = aggregate(embedding*mask_pos, neighborhood, separable)
    s_2 = aggregate(embedding_sum_squares*mask_pos, neighborhood, separable)

    # get distance of each voxel to this label embedding
    distances = (
        embedding_sum_squares*s_0 -
        2.0*tf.reduce_sum(embedding*s_1, axis=0, keep_dims=True) +
        s_2)

    # count pairs
    count_pos = tf.reduce_sum(s_0*mask_pos)
    count_neg = tf.reduce_sum(s_0*mask_neg)

    # compute loss
    loss_pos = tf.reduce_sum(distances*mask_pos)
    loss_neg = tf.reduce_sum(distances*mask_neg)

    return (loss_pos, loss_neg, count_pos, count_neg)

def save_div(a, b, eps=1e-6):
    '''Divide a by b, if b is larger than eps. Otherwise, return a.'''

    return tf.cond(
        tf.greater_equal(b, eps),
        lambda: a/b,
        lambda: a)

def mask_loss_op(
        embedding,
        object_masks,
        background_mask,
        neighborhood,
        separable=False):
    '''Returns a tensorflow op to compute the mask loss.

    The mask loss measures the weighted embedding distance of every voxel to
    nearby voxels of the same object (positive contribution) and other objects
    (negative contribution). Minimizing this loss encourages embedding vectors
    of the same object to be locally similar, and of different objects to be
    locally dissimilar.

    Args:

        embedding (Tensor, shape ``(k, d, h, w)``): A k-dimensional feature
            embedding of points in 3D.

        object_masks (Tensor, shape ``(n, d, h, w)``): Binary masks of the
            ground-truth objects.

        background_mask (Tensor, shape ``(d, h, w)``): Voxels to ignore (e.g.,
            boundary voxels between objects.

        neighborhood (Tensor, shape ``(d, h, w)`` or ``(w,)``): The neighborhood
            to consider to minimize/maximize distance to feature vectors of
            same/other objects. Should be 1D if ``separable`` is ``True``.

        separable (bool, optional): Indicate that the neighborhood is 1D and a
            separable convolution can be used.
    '''

    n, depth, height, width = object_masks.get_shape().as_list()

    # reshape embedding into (k, 1, d, h, w)
    embedding = embedding[:, None, :, :, :]

    # reshape object_masks into (n, 1, d, h, w)
    object_masks = tf.reshape(object_masks, (n, 1, depth, height, width))

    # reshape neighborhood into (d, h, w, 1, 1)
    if separable:
        neighborhood = neighborhood[None,None,:,None,None]
    else:
        neighborhood = neighborhood[:,:,:,None,None]

    # create a foreground mask
    mask_fg = 1.0 - background_mask

    # element-wise squares of the embedding
    embedding_sum_squares = tf.reduce_sum(
        tf.square(embedding),
        axis=0,
        keep_dims=True)

    loss_pos = tf.constant(0, dtype=tf.float32)
    loss_neg = tf.constant(0, dtype=tf.float32)
    count_pos = tf.constant(0, dtype=tf.float32)
    count_neg= tf.constant(0, dtype=tf.float32)

    for i in range(n):

        lp, ln, cp, cn = mask_loss(
            embedding,
            embedding_sum_squares,
            object_masks[i][None,:],
            mask_fg,
            neighborhood,
            separable)

        loss_pos += lp
        loss_neg += ln
        count_pos += cp
        count_neg += cn

    # normalize the loss
    loss_pos = save_div(loss_pos, count_pos)
    loss_neg = save_div(loss_neg, count_neg)

    return loss_pos - loss_neg, loss_pos, loss_neg, count_pos, count_neg
