import pandas as pd
import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
import time
import subprocess
import  os
import re
import time
table=pd.read_csv('table.csv')


def start_test_pb(table,seq,batch):
  tf.reset_default_graph()
  run_metadata = tf.RunMetadata()
  input_pb=table.iloc[seq].tasks

  input_set=["import/"+input_node[0]+":0" for input_node in eval(table.iloc[seq].input_nodes)]
  input_size=[input_node[1]*batch for i,input_node in enumerate(eval(table.iloc[seq].input_nodes))]

  output_node=table.iloc[seq].output_node


  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99)
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:  

    
    with tf.gfile.GFile(input_pb,'rb') as f:  
        frozen_graph = tf.GraphDef()  
        frozen_graph.ParseFromString(f.read())  

    tf.import_graph_def(  
	    frozen_graph,  
	    return_elements=[output_node])
    output_node = "import/"+output_node+":0"
    
   

    
    for step in range(251):
       
        if step==150:
            start_time=time.time()
        
#        precision_for_np=np.float16 if precision=="FP16" else np.float32   
        precision_for_np=np.float32
        def wash(shape):
            return [abs(int(s)) for s in shape]
        input_shapes=[wash(shape) for shape in input_size]
        data_shapes=[np.random.randn(*wash(shape)).astype(precision_for_np) for shape in input_shapes]

        
        feed_dict={input_set[i]:data_shapes[i] for i in range(len(input_set))}

        sess.run(output_node,feed_dict=feed_dict) 
        
        if step==250 :
            sess.run(output_node,feed_dict=feed_dict, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),run_metadata=tf.RunMetadata())
            trace = timeline.Timeline(step_stats=tf.RunMetadata().step_stats)
            with open(input_pb[:-3]+'.json' , 'w') as trace_file:
               trace_file.write(trace.generate_chrome_trace_format())
        if step==249:
            end_time=time.time()
  return   (end_time-start_time)/100



batch_sizes=table['batch_size']
for i in range(len(table)):
      if table.tasks[i].split('.')[-1]=='pb' :
#       if 'FP16.pb'in table.tasks[i]:
#
#         table.loc[i,'step_time']=start_test_pb(table,i,batch_sizes[i])

         result=start_test_pb(table,i,batch_sizes[i])
         print('*'*30)
         print(table.tasks[i])
         print(result)
    

#      if table.tasks[i].split('.')[1]=='uff':
#          table.loc[i,'step_time']=start_test_uff(table,i,[1])
#      if table.tasks[i].split('.')[1]=='onnx':
#          table.loc[i,'step_time']=start_test_onnx(table,i,batch_sizes[i])

table.to_csv('table.csv')

