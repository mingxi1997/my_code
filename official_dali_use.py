import os.path

test_data_root = os.environ['DALI_EXTRA_PATH']

# Caffe LMDB
lmdb_folder = os.path.join(test_data_root, 'db', 'lmdb')

N = 8             # number of GPUs
BATCH_SIZE = 128  # batch size per GPU
ITERATIONS = 32
IMAGE_SIZE = 3

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

class CaffeReadPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus):
        super(CaffeReadPipeline, self).__init__(batch_size, num_threads, device_id)

        self.input = ops.CaffeReader(path = lmdb_folder,
                                     random_shuffle = True, shard_id = device_id, num_shards = num_gpus)
        self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
        self.resize = ops.Resize(device = "gpu",
                                 image_type = types.RGB,
                                 interp_type = types.INTERP_LINEAR)
        self.cmn = ops.CropMirrorNormalize(device = "gpu",
                                            output_dtype = types.FLOAT,
                                            crop = (227, 227),
                                            image_type = types.RGB,
                                            mean = [128., 128., 128.],
                                            std = [1., 1., 1.])
        self.uniform = ops.Uniform(range = (0.0, 1.0))
        self.resize_rng = ops.Uniform(range = (256, 480))

    def define_graph(self):
        inputs, labels = self.input(name="Reader")
        images = self.decode(inputs)
        images = self.resize(images, resize_shorter = self.resize_rng())
        output = self.cmn(images, crop_pos_x = self.uniform(),
                          crop_pos_y = self.uniform())
        return (output, labels)
        
        
    from __future__ import print_function
import numpy as np
from nvidia.dali.plugin.pytorch import DALIGenericIterator

label_range = (0, 999)
pipes = [CaffeReadPipeline(batch_size=BATCH_SIZE, num_threads=2, device_id = device_id, num_gpus = N) for device_id in range(N)]
pipes[0].build()
dali_iter = DALIGenericIterator(pipes, ['data', 'label'], pipes[0].epoch_size("Reader"))
for i, data in enumerate(dali_iter):
    if i >= ITERATIONS:
        break
    # Testing correctness of labels
    for d in data:
        label = d["label"]
        image = d["data"]
        ## labels need to be integers
        assert(np.equal(np.mod(label, 1), 0).all())
        ## labels need to be in range pipe_name[2]
        assert((label >= label_range[0]).all())
        assert((label <= label_range[1]).all())
print("OK")
