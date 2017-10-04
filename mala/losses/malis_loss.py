import tensorflow as tf
import numpy as np
import malis

class MalisWeights(object):

    def __init__(self, output_shape, neighborhood):

        self.output_shape = np.asarray(output_shape)
        self.neighborhood = np.asarray(neighborhood)
        self.edge_list = malis.nodelist_like(self.output_shape, self.neighborhood)

    def get_edge_weights(self, affs, gt_affs, gt_seg):

        assert affs.shape[0] == len(self.neighborhood)

        weights_neg = self.malis_pass(affs, gt_affs, gt_seg, pos=0)
        weights_pos = self.malis_pass(affs, gt_affs, gt_seg, pos=1)

        return (weights_neg, weights_pos)

    def malis_pass(self, affs, gt_affs, gt_seg, pos):

        # in the positive pass (pos==1), we set boundary edges (gt_aff==0) to their true values
        # in the negative pass (pos==0), we set non-boundary edges (gt_aff==1) to their true values
        affs[gt_affs == (1-pos)] = 1-pos

        weights = malis.malis_loss_weights(
            gt_seg.astype(np.uint64).flatten(),
            self.edge_list[0].flatten(),
            self.edge_list[1].flatten(),
            affs.astype(np.float32).flatten(),
            pos)

        weights = weights.reshape((-1,) + tuple(self.output_shape))
        assert weights.shape[0] == len(self.neighborhood)

        # normalize
        weights = weights.astype(np.float32)
        num_pairs = np.sum(weights)
        if num_pairs > 0:
            weights = weights/num_pairs

        return weights

def malis_weights_op(affs, gt_affs, gt_seg, neighborhood, name=None):
    '''Returns a tensorflow op to compute the weights of the MALIS loss. This
    is to be multiplied with an edge-wise loss (e.g., an Euclidean loss).
    '''

    output_shape = gt_seg.get_shape().as_list()

    malis_weights = MalisWeights(output_shape, neighborhood)
    malis_functor = lambda affs, gt_affs, gt_seg, mw=malis_weights: \
        mw.get_edge_weights(affs, gt_affs, gt_seg)

    weights_neg, weights_pos = tf.py_func(
        malis_functor,
        [affs, gt_affs, gt_seg],
        [tf.float32],
        name=name)

    return (weights_neg, weights_pos)

def malis_loss_op(affs, gt_affs, gt_seg, neighborhood, name=None):
    '''Returns a tensorflow op to compute the MALIS loss.'''

    weights_neg, weights_pos = malis_weights_op(affs, gt_affs, gt_seg, neighborhood, name)
    edge_loss_neg = tf.multiply(weights_neg,tf.square(tf.subtract(affs, 0)))
    edge_loss_pos = tf.multiply(weights_pos,tf.square(tf.subtract(affs, 1)))

    return tf.reduce_sum(tf.add(edge_loss_neg,edge_loss_pos))
