import tensorflow as tf  
from tensorflow.python.compiler.tensorrt import trt_convert as trt  
from tensorflow.python.platform import gfile  
#import pdb  
import numpy as np
#from PIL import Image
#import tensorio
import matplotlib.pyplot as plt

input_pb="stylegan2_generator_trainingFalse.opt.pb"
input_node="import/dlatents_in:0"
output_node= "images_out"

step_num=100
shape=(1,1,16,512)
with tf.Session() as sess:  
#load frozen graph
    with tf.gfile.GFile(input_pb,'rb') as f:  
        frozen_graph = tf.GraphDef()  
        frozen_graph.ParseFromString(f.read())  
        # Now you can create a TensorRT inference graph from your  
        # frozen graph:  
    converter = trt.TrtGraphConverter( 
       
	            input_graph_def=frozen_graph,  
	            minimum_segment_size=3,  
	            precision_mode="FP32",  #INT8/FP16/FP32
	            nodes_blacklist=[output_node]) #output nodes  
    trt_graph = converter.convert()  
	#save model
   
    with gfile.FastGFile("tmp.pb",'wb') as f:  
	    f.write(trt_graph.SerializeToString())
	#run graph
    tf.import_graph_def(  
	    trt_graph,  
	    return_elements=[output_node])
    output_node = "import/images_out:0"

    import datetime
    for i in range(5):
        np.random.seed(i)
        x=np.random.rand(*shape).astype(np.float32)*100
        np.save('input{}.npy'.format(i),x)
        print("step {} start  :".format(i),datetime.datetime.now()) 
        out=sess.run(output_node,feed_dict={input_node:x}) #need input data
        np.save('output{}.npy'.format(i),out)
        

