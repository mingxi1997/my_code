import pandas as pd




import tensorflow as tf


import numpy as np
import time
table=pd.read_csv('table.csv')
#for seq in range(len(table.tasks)):

def start_test(table,seq):
  tf.reset_default_graph()
  input_pb=table.iloc[seq].tasks
  input_set=["import/"+input_node[0]+":0" for input_node in eval(table.iloc[seq].input_nodes)]
  input_size=[input_node[1] for input_node in eval(table.iloc[seq].input_nodes)]
#input_node0="import/input_ids:0"
#input_node1="import/input_mask:0"
#input_node2="import/segment_ids:0"
  output_node=table.iloc[seq].output_node

  precision="FP16" if "FP16" in  input_pb else "FP32"

  with tf.Session() as sess:  
#load frozen graph
    with tf.gfile.GFile(input_pb,'rb') as f:  
        frozen_graph = tf.GraphDef()  
        frozen_graph.ParseFromString(f.read())  
        # Now you can create a TensorRT inference graph from your  
        # frozen graph:  
   
	#run graph
    tf.import_graph_def(  
	    frozen_graph,  
	    return_elements=[output_node])
    output_node = "import/"+output_node+":0"
    
   

    print("precision :",precision)
    for step in range(250):
       
        if step==150:
            start_time=time.time()
        
        precision_for_np=np.float16 if precision=="FP16" else np.float32     
        def wash(shape):
            return [abs(int(s)) for s in shape]
        input_shapes=[wash(shape) for shape in input_size]
        data_shapes=[np.random.randn(*wash(shape)).astype(precision_for_np) for shape in input_shapes]
#        
          
        feed_dict={input_set[i]:data_shapes[i] for i in range(len(input_set))}
        
        sess.run(output_node,feed_dict=feed_dict) #need input data

    end_time=time.time()
  return   (end_time-start_time)/100
for i in range(len(table)):
    table.loc[i,'step_time']=start_test(table,i)

table.to_csv('table.csv')









