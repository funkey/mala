import logging
from gunpowder import BatchFilter, Volume
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import convolve
import numpy as np
import time

logger = logging.getLogger(__name__)

class AddLocalShapeDescriptor(BatchFilter):
    '''Create a local segmentation shape discriptor to each voxel.

    Args:

        segmentation (:class:`VolumeType`): The volume storing the segmentation
            to use.

        descriptor (:class:`VolumeType`): The volume of the shape descriptor to
            generate.

        sigma (float or tuple of float): The context to consider to compute
            the shape descriptor in world units. This will be the standard
            deviation of a Gaussian kernel or the radius of the sphere.

        mode (string): Either ``gaussian`` or ``sphere``. Specifies how to
            accumulate local statistics: ``gaussian`` uses Gaussian convolution
            to compute a weighed average of statistics inside an object.
            ``sphere`` accumulates values in a sphere.
    '''

    def __init__(self, segmentation, descriptor, sigma, mode='gaussian'):

        self.segmentation = segmentation
        self.descriptor = descriptor
        try:
            self.sigma = tuple(sigma)
        except:
            self.sigma = (sigma,)*3
        self.mode = mode
        self.voxel_size = None
        self.context = None
        self.skip = False

    def setup(self):

        spec = self.spec[self.segmentation].copy()
        spec.dtype = np.float32

        self.voxel_size = spec.voxel_size
        self.provides(self.descriptor, spec)

        if self.mode == 'gaussian':
            self.context = tuple(s*3.0 for s in self.sigma)
        elif self.mode == 'sphere':
            self.context = tuple(self.sigma)
        else:
            raise RuntimeError("Unkown mode %s"%mode)

    def prepare(self, request):

        if self.descriptor in request:
            del request[self.descriptor]
            self.skip = False
        else:
            self.skip = True

        # increase segmentation ROI to fit Gaussian
        grown_roi = request[self.segmentation].roi.grow(
            self.context,
            self.context)
        request[self.segmentation].roi = grown_roi

    def process(self, batch, request):

        if self.skip:
            return

        dims = len(self.voxel_size)

        assert dims == 3, "AddLocalShapeDescriptor only works on 3D volumes."

        segmentation_volume = batch.volumes[self.segmentation]

        descriptor = self.__get_descriptor(segmentation_volume.data)

        descriptor_spec = self.spec[self.descriptor].copy()
        descriptor_spec.roi = segmentation_volume.spec.roi.copy()
        descriptor_volume = Volume(descriptor, descriptor_spec)

        # crop segmentation and descriptor to original segmentation ROI
        request_roi = request[self.segmentation].roi

        descriptor_volume = descriptor_volume.crop(request_roi)
        segmentation_volume = segmentation_volume.crop(request_roi)

        batch.volumes[self.segmentation] = segmentation_volume
        batch.volumes[self.descriptor] = descriptor_volume

    def __get_descriptor(self, segmentation):

        sigma_voxel = tuple(s/v for s, v in zip(self.sigma, self.voxel_size))
        logger.debug("Sigma in voxels: %s", sigma_voxel)

        depth, height, width = segmentation.shape

        counts = np.zeros((depth, height, width), dtype=np.float32)
        mean_offsets = np.zeros((3, depth, height, width), dtype=np.float32)
        variances = np.zeros((3, depth, height, width), dtype=np.float32)
        pearsons = np.zeros((3, depth, height, width), dtype=np.float32)

        for label in np.unique(segmentation):

            if label == 0:
                continue

            logger.debug("Creating shape descriptors for label %d", label)

            mask = (segmentation==label).astype(np.float32)

            # voxel coordinates in world units
            # TODO: don't recreate
            logger.debug("Create meshgrid...")
            coords = np.array(
                np.meshgrid(
                    np.arange(0, depth*self.voxel_size[0], self.voxel_size[0]),
                    np.arange(0, height*self.voxel_size[1], self.voxel_size[1]),
                    np.arange(0, width*self.voxel_size[2], self.voxel_size[2]),
                    indexing='ij'),
                dtype=np.float32)

            # mask for object
            coords[:, mask==0] = 0

            # number of inside voxels
            logger.debug("Counting inside voxels...")
            start = time.time()
            count = self.__aggregate(mask, sigma_voxel, self.mode)
            count[mask==0] = 0
            counts += count
            # avoid division by zero
            count[count==0] = 1
            logger.debug("%f seconds", time.time() - start)

            # mean
            logger.debug("Computing mean position of inside voxels...")
            start = time.time()
            mean = np.array([self.__aggregate(coords[d], sigma_voxel, self.mode) for d in range(3)])
            mean /= count
            logger.debug("%f seconds", time.time() - start)

            logger.debug("Computing offset of mean position...")
            start = time.time()
            mean_offset = mean - coords

            mean_offset[:, mask==0] = 0
            mean_offsets += mean_offset

            # covariance
            logger.debug("Computing covariance...")
            coords_outer = self.__outer_product(coords)
            covariance = np.array([
                self.__aggregate(
                    coords_outer[d],
                    sigma_voxel,
                    self.mode)
                for d in range(9)])
            covariance /= count
            covariance -= self.__outer_product(mean)
            logger.debug("%f seconds", time.time() - start)

            # remove duplicate entries in covariance
            # 0 1 2
            # 3 4 5
            # 6 7 8
            # variances of z, y, x coordinates
            variance = covariance[[0, 4, 8]]
            # Pearson coefficients of zy, zx, yx
            pearson = covariance[[1, 2, 5]]

            # normalize Pearson correlation coefficient
            variance[variance<1e-3] = 1e-3 # numerical stability
            pearson[0] /= np.sqrt(variance[0])*np.sqrt(variance[1])
            pearson[1] /= np.sqrt(variance[0])*np.sqrt(variance[2])
            pearson[2] /= np.sqrt(variance[1])*np.sqrt(variance[2])

            # normalize variances to interval [0, 1]
            variance[0] /= self.sigma[0]**2
            variance[1] /= self.sigma[1]**2
            variance[2] /= self.sigma[2]**2

            variance[:, mask==0] = 0
            pearson[:, mask==0] = 0
            variances += variance
            pearsons += pearson

        return np.concatenate([mean_offsets, variances, pearsons, counts[None,:]])

    def __make_sphere(self, radius):

        logger.debug("Creating sphere with radius %d...", radius)

        r2 = np.arange(-radius, radius)**2
        dist2 = r2[:, None, None] + r2[:, None] + r2
        return (dist2 <= radius**2).astype(np.float32)

    def __aggregate(self, array, sigma, mode='gaussian'):

        if mode == 'gaussian':

            return gaussian_filter(
                array,
                sigma=sigma,
                mode='constant',
                cval=0.0,
                truncate=3.0)

        elif mode == 'sphere':

            radius = sigma[0]
            for d in range(len(sigma)):
                assert radius == sigma[d], (
                    "For mode 'sphere', only isotropic sigma is allowed.")

            sphere = self.__make_sphere(radius)
            return convolve(array, sphere, mode='constant', cval=0.0)

        else:
            raise RuntimeError("Unknown mode %s"%mode)

    def __outer_product(self, array):
        '''Computes the unique values of the outer products of the first dimension
        of ``array``. If ``array`` has shape ``(k, d, h, w)``, for example, the
        output will be of shape ``(k*(k+1)/2, d, h, w)``.
        '''
        k = array.shape[0]
        outer = np.einsum('i...,j...->ij...', array, array)
        return outer.reshape((k**2,)+array.shape[1:])
