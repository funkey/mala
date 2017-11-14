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

    b, k, d, h, w = stats.get_shape().as_list()
    stats = tf.reshape(stats, [k, 1, d, h, w])

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

    return tf.reshape(agg, [1, k, d, h, w])

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
        2.0*tf.reduce_sum(embedding*s_1, axis=1, keep_dims=True) +
        s_2)

    # combine to total loss (positive distance within object, negative outside)
    return distances*(2.0*mask - 1.0)

def mask_loss_op(
        embedding,
        gt_seg,
        neighborhood,
        separable=False):
    # TODO: check comments on dimensions
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

        sigma (float): The standard deviation of a Gaussian used to weigh the
            embedding distances as a function of spatial distance.
    '''

    # create a neighborhood kernel
    # TODO: create Gaussian
    neighborhood = neighborhood[:,:,:,None,None]

    # element-wise squares of the embedding
    embedding_sum_squares = tf.reduce_sum(
        tf.square(embedding),
        axis=1,
        keep_dims=True)

    # list of all labels
    labels = tf.unique(tf.reshape(gt_seg, [-1]))[0]
    num_labels = tf.size(labels)

    # iterate over all labels
    i = tf.constant(0)

    loss = tf.to_float(tf.zeros_like(gt_seg))

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
    _, loss = tf.while_loop(iterate, add_loss, [i, loss])

    return tf.reduce_sum(loss), loss
