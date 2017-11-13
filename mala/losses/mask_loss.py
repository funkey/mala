import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

def aggregate(stats, neighborhood):
    '''Convolve stats, such that each point contains a weighted sum of the
    stats around it.'''

    return tf.nn.convolution(
        stats,
        neighborhood,
        padding='SAME',
        data_format='NCDHW')

def label_loss(
    label,
    embedding,
    embedding_sum_squares,
    gt_seg,
    neighborhood):
    '''
    Compute the mask loss for the given label only.
    '''

    # create the label mask
    mask = tf.to_float(tf.equal(gt_seg, label))

    # aggregate s_0, s_1, and s_2 scores from mask, embedding, and
    # embedding_sum_squares
    s_0 = aggregate(mask, neighborhood)
    s_1 = aggregate(embedding*mask, neighborhood)
    s_2 = aggregate(embedding_sum_squares*mask, neighborhood)

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
        neighborhood):
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
            neighborhood)]
    _, loss = tf.while_loop(iterate, add_loss, [i, loss])

    return tf.reduce_sum(loss), loss
