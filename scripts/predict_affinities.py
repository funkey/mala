import sys
from gunpowder import *
from gunpowder.tensorflow import Predict
import os
import json
import numpy as np

data_dir = '../01_data'

def predict_affinities(setup, iteration, sample):

    checkpoint = os.path.join('../02_train', setup, 'unet_checkpoint_%d'%iteration)
    with open(os.path.join('../02_train', setup, 'net_io_names.json'), 'r') as f:
        net_io_names = json.load(f)

    voxel_size = Coordinate((40, 4, 4))
    input_size = Coordinate((84, 268, 268))*voxel_size
    output_size = Coordinate((56,56,56))*voxel_size
    context = (input_size - output_size)/2

    register_volume_type('PRED_AFFINITIES')

    chunk_request = BatchRequest()
    chunk_request.add(VolumeTypes.RAW, input_size)
    chunk_request.add(VolumeTypes.PRED_AFFINITIES, output_size)
    chunk_request[VolumeTypes.RAW].voxel_size = voxel_size
    chunk_request[VolumeTypes.PRED_AFFINITIES].voxel_size = voxel_size

    source = (
        Hdf5Source(
            os.path.join(data_dir, sample + '.hdf'),
            datasets = { VolumeTypes.RAW: 'volumes/raw'}) +
        Normalize() +
        Pad({ VolumeTypes.RAW: (4000, 400, 400) }))

    with build(source):
        raw_spec = source.spec[VolumeTypes.RAW]

    pipeline = (
        source +
        IntensityScaleShift(2, -1) +
        ZeroOutConstSections() +
        Predict(
            checkpoint,
            inputs = {
                net_io_names['raw']: VolumeTypes.RAW,
            },
            outputs = {
                net_io_names['affs']: VolumeTypes.PRED_AFFINITIES,
            },
            volume_specs = {
                VolumeTypes.PRED_AFFINITIES: VolumeSpec(
                    roi=raw_spec.roi,
                    voxel_size=raw_spec.voxel_size,
                    dtype=np.float32
                )
            }) +
        PrintProfilingStats() +
        Chunk(chunk_request) +
        Snapshot({
                VolumeTypes.RAW: 'volumes/raw',
                VolumeTypes.PRED_AFFINITIES: 'volumes/predicted_affs',
            },
            dataset_dtypes={
                VolumeTypes.PRED_AFFINITIES: np.float32,
            },
            every=1,
            output_dir=os.path.join('processed', setup, '%d'%iteration),
            output_filename=sample+'.hdf')
    )

    with build(pipeline):

        raw_spec = source.spec[VolumeTypes.RAW].copy()
        aff_spec = raw_spec.copy()
        aff_spec.roi = raw_spec.roi.grow(-context, -context)

        whole_request = BatchRequest({
                VolumeTypes.RAW: raw_spec,
                VolumeTypes.PRED_AFFINITIES: aff_spec
            })

        print("Requesting " + str(whole_request) + " in chunks of " + str(chunk_request))

        pipeline.request_batch(whole_request)

if __name__ == "__main__":
    setup = sys.argv[1]
    iteration = int(sys.argv[2])
    sample = sys.argv[3]
    set_verbose(False)
    predict_affinities(setup, iteration, sample)
