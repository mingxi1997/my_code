
#Let us start from defining some global constants

#DALI_EXTRA_PATH environment variable should point to the place where data from DALI extra repository is downloaded. Please make sure that the proper release tag is checked out.

#In [1]:
import os.path

test_data_root = os.environ['DALI_EXTRA_PATH']

# MXNet RecordIO
db_folder = os.path.join(test_data_root, 'db', 'recordio/')

# Caffe LMDB
lmdb_folder = os.path.join(test_data_root, 'db', 'lmdb')

# image dir with plain jpeg files
image_dir = "../../data/images"

# TFRecord
tfrecord = os.path.join(test_data_root, 'db', 'tfrecord', 'train')
tfrecord_idx = "idx_files/train.idx"
tfrecord2idx_script = "tfrecord2idx"

N = 8             # number of GPUs
BATCH_SIZE = 128  # batch size per GPU
ITERATIONS = 32
IMAGE_SIZE = 3
#Create idx file by calling tfrecord2idx script

#In [2]:
from subprocess import call
import os.path

if not os.path.exists("idx_files"):
    os.mkdir("idx_files")

if not os.path.isfile(tfrecord_idx):
    call([tfrecord2idx_script, tfrecord, tfrecord_idx])
Let us define:

#common part of pipeline, other pipelines will inherit it
#In [3]:
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

class CommonPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(CommonPipeline, self).__init__(batch_size, num_threads, device_id)

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

    def base_define_graph(self, inputs, labels):
        images = self.decode(inputs)
        images = self.resize(images, resize_shorter = self.resize_rng())
        output = self.cmn(images, crop_pos_x = self.uniform(),
                          crop_pos_y = self.uniform())
        return (output, labels)
#MXNetReaderPipeline
#In [4]:
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

class MXNetReaderPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus):
        super(MXNetReaderPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.MXNetReader(path = [db_folder+"train.rec"], index_path=[db_folder+"train.idx"],
                                     random_shuffle = True, shard_id = device_id, num_shards = num_gpus)

    def define_graph(self):
        images, labels = self.input(name="Reader")
        return self.base_define_graph(images, labels)
#CaffeReadPipeline
#In [5]:
class CaffeReadPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus):
        super(CaffeReadPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.CaffeReader(path = lmdb_folder,
                                     random_shuffle = True, shard_id = device_id, num_shards = num_gpus)

    def define_graph(self):
        images, labels = self.input(name="Reader")
        return self.base_define_graph(images, labels)
#FileReadPipeline
#In [6]:
class FileReadPipeline(CommonPipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus):
            super(FileReadPipeline, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.FileReader(file_root = image_dir)

        def define_graph(self):
            images, labels = self.input(name="Reader")
            return self.base_define_graph(images, labels)
#TFRecordPipeline
#In [7]:
import nvidia.dali.tfrecord as tfrec

class TFRecordPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus):
        super(TFRecordPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.TFRecordReader(path = tfrecord, 
                                        index_path = tfrecord_idx,
                                        features = {"image/encoded" : tfrec.FixedLenFeature((), tfrec.string, ""),
                                                    "image/class/label": tfrec.FixedLenFeature([1], tfrec.int64,  -1)
                                        })

    def define_graph(self):
        inputs = self.input(name="Reader")
        images = inputs["image/encoded"]
        labels = inputs["image/class/label"]
        return self.base_define_graph(images, labels)
#Let us create pipelines and pass them to PyTorch generic iterator

#In [8]:
from __future__ import print_function
import numpy as np
from nvidia.dali.plugin.pytorch import DALIGenericIterator

pipe_types = [[MXNetReaderPipeline, (0, 999)], 
              [CaffeReadPipeline, (0, 999)],
              [FileReadPipeline, (0, 1)], 
              [TFRecordPipeline, (1, 1000)]]
for pipe_t in pipe_types:
    pipe_name, label_range = pipe_t
    print ("RUN: "  + pipe_name.__name__)
    pipes = [pipe_name(batch_size=BATCH_SIZE, num_threads=2, device_id = device_id, num_gpus = N) for device_id in range(N)]
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
    print("OK : " + pipe_name.__name__)
