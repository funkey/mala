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
    #   This seems very wateful.
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
    gt_seg,
    neighborhood,
    separable):
    '''
    Compute the mask loss for the given label only.
    '''

    # create the label mask
    mask = tf.to_float(tf.equal(gt_seg, label))

    # aggregate s_0, s_1, and s_2 scores from mask, embedding, and
    # embedding_sum_squares
    s_0 = aggregate(mask, neighborhood, separable)
    s_1 = aggregate(embedding*mask, neighborhood, separable)
    s_2 = aggregate(embedding_sum_squares*mask, neighborhood, separable)

    # get distance of each voxel to this label embedding
    distances = (
        embedding_sum_squares*s_0 -
        2.0*tf.reduce_sum(embedding*s_1, axis=0, keep_dims=True) +
        s_2)

    # combine to total loss (positive distance within object, negative outside)
    label_loss = distances*(2.0*mask - 1.0)

    # strip batch and channel dimensions
    return label_loss[0, 0]

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

    loss = tf.to_float(tf.zeros_like(gt_seg))

    # reshape embedding into (k, 1, d, h, w)
    embedding = embedding[:, None, :, :, :]

    # reshape gt_seg into (1, 1, d, h, w)
    gt_seg = gt_seg[None, None, :, :, :]

    # reshape neighborhood into (d, h, w, 1, 1)
    if separable:
        neighborhood = neighborhood[None,None,:,None,None]
    else:
        neighborhood = neighborhood[:,:,:,None,None]

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

    # just a strange way to write a for loop: loop over all labels in gt_seg,
    # compute the 'label_loss' and add it to 'loss'
    iterate = lambda i, loss : tf.less(i, num_labels)
    add_loss = lambda i, loss : [
        i + 1,
        loss + label_loss(
            labels[i],
            embedding,
            embedding_sum_squares,
            gt_seg,
            neighborhood,
            separable)]
    _, loss = tf.while_loop(
        iterate,
        add_loss,
        [i, loss],
        swap_memory=swap_memory)

    return tf.reduce_sum(loss), loss
