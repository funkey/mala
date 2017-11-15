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

def label_loss(
    label,
    embedding,
    embedding_sum_squares,
    mask_fg,
    gt_seg,
    neighborhood,
    separable):
    '''
    Compute the mask loss for the given label only. Returns postive and negative
    contributions and the count of positive and negative pairrs (weighted by
    ``neighborhood``).
    '''

    # create the label mask
    mask_pos = tf.to_float(tf.equal(gt_seg, label))
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

def add_label_loss(
    loss_pos, loss_neg,
    count_pos, count_neg,
    label,
    embedding,
    embedding_sum_squares,
    mask_fg,
    gt_seg,
    neighborhood,
    separable):

    lp, ln, cp, cn = label_loss(
        label,
        embedding,
        embedding_sum_squares,
        mask_fg,
        gt_seg,
        neighborhood,
        separable)

    return (loss_pos + lp, loss_neg + ln, count_pos + cp, count_neg + cn)

def add_fg_label_loss(
    i,
    loss_pos, loss_neg,
    count_pos, count_neg,
    label,
    embedding,
    embedding_sum_squares,
    mask_fg,
    gt_seg,
    neighborhood,
    separable):
    '''Helper for tf.while_loop.'''

    # ignore background label
    lp, ln, cp, cn = tf.cond(
        tf.equal(label, 0),
        lambda: (loss_pos, loss_neg, count_pos, count_neg),
        lambda: add_label_loss(
            loss_pos, loss_neg,
            count_pos, count_neg,
            label,
            embedding,
            embedding_sum_squares,
            mask_fg,
            gt_seg,
            neighborhood,
            separable))

    return (i + 1, lp, ln, cp, cn)

def save_div(a, b, eps=1e-6):
    '''Divide a by b, if b is larger than eps. Otherwise, return a.'''

    return tf.cond(
        tf.greater_equal(b, 1e-6),
        lambda: a/b,
        lambda: a)

def mask_loss_op(
        embedding,
        gt_seg,
        neighborhood,
        separable=False,
        swap_memory=False):
    '''Returns a tensorflow op to compute the mask loss.

    The mask loss measures the weighted embedding distance of every voxel to
    nearby voxels of the same object (positive contribution) and other objects
    (negative contribution). Minimizing this loss encourages embedding vectors
    of the same object to be locally similar, and of different objects to be
    locally dissimilar.

    Args:

        embedding (Tensor, shape ``(k, d, h, w)``): A k-dimensional feature
            embedding of points in 3D.

        gt_seg (Tensor, shape ``(d, h, w)``): The ground-truth labels of the
            points.

        neighborhood (Tensor, shape ``(d, h, w)`` or ``(w,)``): The neighborhood
            to consider to minimize/maximize distance to feature vectors of
            same/other objects. Should be 1D if ``separable`` is ``True``.

        separable (bool, optional): Indicate that the neighborhood is 1D and a
            separable convolution can be used.

        swap_memory (bool, optional): Since convolutions are performed for ever
            object in ``gt_seg``, this operator can exceed the memory available.
            This option will swap memory between the CPU and GPU for each label
            to avoid running out of memory.
    '''

    # reshape embedding into (k, 1, d, h, w)
    embedding = embedding[:, None, :, :, :]

    # reshape gt_seg into (1, 1, d, h, w)
    gt_seg = gt_seg[None, None, :, :, :]

    # reshape neighborhood into (d, h, w, 1, 1)
    if separable:
        neighborhood = neighborhood[None,None,:,None,None]
    else:
        neighborhood = neighborhood[:,:,:,None,None]

    # create a foreground mask
    mask_fg = tf.to_float(tf.not_equal(gt_seg, 0))

    # element-wise squares of the embedding
    embedding_sum_squares = tf.reduce_sum(
        tf.square(embedding),
        axis=0,
        keep_dims=True)

    # list of all labels
    labels = tf.unique(tf.reshape(gt_seg, [-1]))[0]
    num_labels = tf.size(labels)

    # iterate over all labels
    i = tf.constant(0)

    loss_pos = tf.constant(0, dtype=tf.float32)
    loss_neg = tf.constant(0, dtype=tf.float32)
    count_pos = tf.constant(0, dtype=tf.float32)
    count_neg= tf.constant(0, dtype=tf.float32)

    # just a strange way to write a for loop: loop over all labels in gt_seg,
    # compute the 'label_loss' and add it to 'loss_pos' and 'loss_neg'
    test = lambda i, lp, ln, cp, cn: tf.less(i, num_labels)
    body = lambda i, lp, ln, cp, cn: add_fg_label_loss(
        i,
        lp, ln,
        cp, cn,
        labels[i],
        embedding,
        embedding_sum_squares,
        mask_fg,
        gt_seg,
        neighborhood,
        separable)
    _, loss_pos, loss_neg, count_pos, count_neg = tf.while_loop(
        test,
        body,
        [i, loss_pos, loss_neg, count_pos, count_neg],
        swap_memory=swap_memory)

    # normalize the loss
    loss_pos = save_div(loss_pos, count_pos)
    loss_neg = save_div(loss_neg, count_neg)

    return loss_pos - loss_neg, loss_pos, loss_neg, count_pos, count_neg
