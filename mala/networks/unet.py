import tensorflow as tf

def conv_pass(
        fmaps_in,
        kernel_sizes,
        num_fmaps,
        activation='relu',
        name='conv_pass',
        fov=(1, 1, 1),
        voxel_size=(1, 1, 1)):
    '''Create a convolution pass::

        f_in --> f_1 --> ... --> f_n

    where each ``-->`` is a convolution followed by a (non-linear) activation
    function. One convolution will be performed for each entry in
    ``kernel_sizes``. Each convolution will decrease the size of the feature
    maps by ``kernel_size-1``.

    Args:

        f_in:

            The input tensor of shape ``(batch_size, channels, depth, height, width)``.

        kernel_sizes:

            Sizes of the kernels to use. Forwarded to tf.layers.conv3d.

        num_fmaps:

            The number of feature maps to produce with each convolution.

        activation:

            Which activation to use after a convolution. Accepts the name of any
            tensorflow activation function (e.g., ``relu`` for ``tf.nn.relu``).

        name:

            Base name for the conv layer.

        fov:

            Field of view of fmaps_in, in physical units.

        voxel_size:

            Size of a voxel in the input data, in physical units.

    Returns:

        (fmaps, fov):

            The feature maps after the last convolution, and a tuple
            representing the field of view.
    '''

    fmaps = fmaps_in
    if activation is not None:
        activation = getattr(tf.nn, activation)

    for i, kernel_size in enumerate(kernel_sizes):

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]*len(voxel_size)

        fov = tuple(f + (k - 1)*vs for f, k, vs in zip(fov, kernel_size, voxel_size))
        fmaps = tf.layers.conv3d(
            inputs=fmaps,
            filters=num_fmaps,
            kernel_size=kernel_size,
            padding='valid',
            data_format='channels_first',
            activation=activation,
            name=name + '_%i'%i)

    return fmaps, fov

def downsample(
        fmaps_in,
        factors,
        name='down',
        voxel_size=(1, 1, 1)):

    voxel_size = tuple(vs*fac for vs, fac in zip(voxel_size, factors))
    fmaps = tf.layers.max_pooling3d(
        fmaps_in,
        pool_size=factors,
        strides=factors,
        padding='valid',
        data_format='channels_first',
        name=name)

    return fmaps, voxel_size

def upsample(
        fmaps_in,
        factors,
        num_fmaps,
        activation='relu',
        name='up',
        voxel_size=(1, 1, 1)):

    voxel_size = tuple(vs/fac for vs, fac in zip(voxel_size, factors))
    if activation is not None:
        activation = getattr(tf.nn, activation)

    fmaps = tf.layers.conv3d_transpose(
        fmaps_in,
        filters=num_fmaps,
        kernel_size=factors,
        strides=factors,
        padding='valid',
        data_format='channels_first',
        activation=activation,
        name=name)

    return fmaps, voxel_size

def crop_zyx(fmaps_in, shape):
    '''Crop only the spatial dimensions to match shape.

    Args:

        fmaps_in:

            The input tensor.

        shape:

            A list (not a tensor) with the requested shape [_, _, z, y, x].
    '''

    in_shape = fmaps_in.get_shape().as_list()

    offset = [
        0, # batch
        0, # channel
        (in_shape[2] - shape[2])//2, # z
        (in_shape[3] - shape[3])//2, # y
        (in_shape[4] - shape[4])//2, # x
    ]
    size = [
        in_shape[0],
        in_shape[1],
        shape[2],
        shape[3],
        shape[4],
    ]

    fmaps = tf.slice(fmaps_in, offset, size)

    return fmaps

def unet(
        fmaps_in,
        num_fmaps,
        fmap_inc_factors,
        downsample_factors,
        kernel_size_down=None,
        kernel_size_up=None,
        activation='relu',
        layer=0,
        fov=(1, 1, 1),
        voxel_size=(1, 1, 1)):
    '''Create a U-Net::

        f_in --> f_left --------------------------->> f_right--> f_out
                    |                                   ^
                    v                                   |
                 g_in --> g_left ------->> g_right --> g_out
                             |               ^
                             v               |
                                   ...

    where each ``-->`` is a convolution pass (see ``conv_pass``), each `-->>` a
    crop, and down and up arrows are max-pooling and transposed convolutions,
    respectively.

    The U-Net expects tensors to have shape ``(batch=1, channels, depth, height,
    width)``.

    This U-Net performs only "valid" convolutions, i.e., sizes of the feature
    maps decrease after each convolution.

    Args:

        fmaps_in:

            The input tensor.

        num_fmaps:

            The number of feature maps in the first layer. This is also the
            number of output feature maps.

        fmap_inc_factors:

            By how much to multiply the number of feature maps between layers.
            If layer 0 has ``k`` feature maps, layer ``l`` will have
            ``k*fmap_inc_factor**l``.

        downsample_factors:

            List of lists ``[z, y, x]`` to use to down- and up-sample the
            feature maps between layers.

        kernel_size_down (optional):

            List of lists of kernel sizes. The number of sizes in a list
            determines the number of convolutional layers in the corresponding
            level of the build on the left side. Kernel sizes can be given as
            tuples or integer. If not given, each convolutional pass will
            consist of two 3x3x3 convolutions.

        kernel_size_up (optional):

            List of lists of kernel sizes. The number of sizes in a list
            determines the number of convolutional layers in the corresponding
            level of the build on the right side. Within one of the lists going
            from left to right. Kernel sizes can be given as tuples or integer.
            If not given, each convolutional pass will consist of two 3x3x3
            convolutions.

        activation:

            Which activation to use after a convolution. Accepts the name of any
            tensorflow activation function (e.g., ``relu`` for ``tf.nn.relu``).

        layer:

            Used internally to build the U-Net recursively.
        fov:

            Initial field of view in physical units

        voxel_size:

            Size of a voxel in the input data, in physical units
    '''

    prefix = "    "*layer
    print(prefix + "Creating U-Net layer %i"%layer)
    print(prefix + "f_in: " + str(fmaps_in.shape))

    if isinstance(fmap_inc_factors, int):
        fmap_inc_factors = [fmap_inc_factors]*len(downsample_factors)

    # by default, create 2 3x3x3 convolutions per layer
    if kernel_size_down is None:
        kernel_size_down = [ [3, 3] ]*(len(downsample_factors) + 1)
    if kernel_size_up is None:
        kernel_size_up = [ [3, 3] ]*(len(downsample_factors) + 1)

    assert (
        len(fmap_inc_factors) ==
        len(downsample_factors) ==
        len(kernel_size_down) - 1 ==
        len(kernel_size_up) - 1)

    # convolve
    f_left, fov = conv_pass(
        fmaps_in,
        kernel_sizes=kernel_size_down[layer],
        num_fmaps=num_fmaps,
        activation=activation,
        name='unet_layer_%i_left'%layer,
        fov=fov,
        voxel_size=voxel_size)

    # last layer does not recurse
    bottom_layer = (layer == len(downsample_factors))
    if bottom_layer:
        print(prefix + "bottom layer")
        print(prefix + "f_out: " + str(f_left.shape))
        return f_left, fov, voxel_size

    # downsample
    g_in, voxel_size = downsample(
        f_left,
        downsample_factors[layer],
        'unet_down_%i_to_%i'%(layer, layer + 1),
        voxel_size=voxel_size)

    # recursive U-net
    g_out, fov, voxel_size = unet(
        g_in,
        num_fmaps=num_fmaps*fmap_inc_factors[layer],
        fmap_inc_factors=fmap_inc_factors,
        downsample_factors=downsample_factors,
        kernel_size_down=kernel_size_down,
        kernel_size_up=kernel_size_up,
        activation=activation,
        layer=layer+1,
        fov=fov,
        voxel_size=voxel_size)

    print(prefix + "g_out: " + str(g_out.shape))

    # upsample
    g_out_upsampled, voxel_size = upsample(
        g_out,
        downsample_factors[layer],
        num_fmaps,
        activation=activation,
        name='unet_up_%i_to_%i'%(layer + 1, layer),
        voxel_size=voxel_size)

    print(prefix + "g_out_upsampled: " + str(g_out_upsampled.shape))

    # copy-crop
    f_left_cropped = crop_zyx(f_left, g_out_upsampled.get_shape().as_list())

    print(prefix + "f_left_cropped: " + str(f_left_cropped.shape))

    # concatenate along channel dimension
    f_right = tf.concat([f_left_cropped, g_out_upsampled], 1)

    print(prefix + "f_right: " + str(f_right.shape))

    # convolve
    f_out, fov = conv_pass(
        f_right,
        kernel_sizes=kernel_size_up[layer],
        num_fmaps=num_fmaps,
        name='unet_layer_%i_right'%layer,
        fov=fov,
        voxel_size=voxel_size)

    print(prefix + "f_out: " + str(f_out.shape))

    return f_out, fov, voxel_size

if __name__ == "__main__":

    raw = tf.placeholder(tf.float32, shape=(43, 430, 430))
    raw_batched = tf.reshape(raw, (1, 1,) + (43, 430, 430))

    model, ll_fov, vx = unet(
        raw_batched,
        12, 6,
        [[1, 3, 3], [1, 3, 3], [3, 3, 3]],
        [
            [(1, 3, 3), (1, 3, 3)],
            [(1, 3, 3), (1, 3, 3)],
            [(3, 3, 3), (3, 3, 3)],
            [(3, 3, 3), (3, 3, 3)]
        ],
        [
            [(1, 3, 3), (1, 3, 3)],
            [(1, 3, 3), (1, 3, 3)],
            [(3, 3, 3), (3, 3, 3)],
            [(3, 3, 3), (3, 3, 3)]
        ],
        voxel_size=(10, 1, 1),
        fov=(10, 1, 1))

    output, full_fov = conv_pass(
        model,
        kernel_sizes=[(1, 1, 1)],
        num_fmaps=1,
        activation=None,
        fov=ll_fov,
        voxel_size=vx)

    tf.train.export_meta_graph(filename='build.meta')

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        tf.summary.FileWriter('.', graph=tf.get_default_graph())

    print(model.shape)
