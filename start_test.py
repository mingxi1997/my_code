import pandas as pd
import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
import time
import subprocess
import  os

table=pd.read_csv('table.csv')
run_metadata = tf.RunMetadata()

def start_test_pb(table,seq,batch):
  tf.reset_default_graph()
  input_pb=table.iloc[seq].tasks

  input_set=["import/"+input_node[0]+":0" for input_node in eval(table.iloc[seq].input_nodes)]
  input_size=[input_node[1]*batch[i] for i,input_node in enumerate(eval(table.iloc[seq].input_nodes))]

  output_node=table.iloc[seq].output_node



  with tf.Session() as sess:  

    
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
            sess.run(output_node,feed_dict=feed_dict, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),run_metadata=run_metadata)
            trace = timeline.Timeline(step_stats=run_metadata.step_stats)
            with open(input_pb[:-3]+'.json' , 'w') as trace_file:
               trace_file.write(trace.generate_chrome_trace_format())
        if step==249:
            end_time=time.time()
  return   (end_time-start_time)/100

logs_dir = './test/'

def start_test_uff(table, seq, batches):
    uff_name = table.iloc[seq].tasks
    input_nodes_str = table.iloc[seq].input_nodes
    output_nodes_name = table.iloc[seq].output_node
    for batch in batches:
        input_nodes_list = eval(input_nodes_str)  # str=>list
        input_params = ""  # concat uffInput params
        for node in input_nodes_list:
            input_params += " --input="+node[0] + ","
            node_size = node[1]
            node_size = node_size[1:]
            for size in node_size:
                input_params += size + ","
            input_params = input_params[:-1]  # remove redundent comma
            command = './trtexec --uff={} --output={}{} --batch={}'.format(uff_name, output_nodes_name, input_params, batch)
            print('*'*30)
            print(command)
            log = subprocess.run(command, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
            # write log
            log_file = os.path.join(logs_dir, os.path.basename(uff_name).split('.')[0] + "_uff.txt")
            if os.path.isfile(log_file):
                os.remove(log_file)
            with open(log_file, 'w') as f:
                f.write(log)
    return -1


def start_test_onnx(table ,seq,batch):
    pb_name = table.iloc[seq].tasks
    input_nodes_str = table.iloc[seq].input_nodes
    input_nodes_names = eval(input_nodes_str)
    shapes=' '.join(["--shapes=\'"+n[0]+":0\':"+ 'x'.join(str(abs(int(n[1])))*batch[i]) for i,n in enumerate(input_nodes_names)])
    onnx_name = os.path.basename(pb_name).split('.')[0] + ".onnx"
    command = './trtexec --onnx={} {} --fp16'.format(onnx_name, shapes)
    print(command)
    log_onnxtest = subprocess.run(command, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
    # write log
    log_file = os.path.join(logs_dir, os.path.basename(pb_name).split('.')[0] + "_onnx.txt")
    if os.path.isfile(log_file):
        os.remove(log_file)
    with open(log_file, 'w') as f:
        f.write(log_onnxtest)
    return -1

for i in range(len(table)):
      if table.tasks[i].split('.')[1]=='pb':
          table.loc[i,'step_time']=start_test_pb(table,i,[1])
      if table.tasks[i].split('.')[1]=='uff':
          table.loc[i,'step_time']=start_test_uff(table,i,[1])
      if table.tasks[i].split('.')[1]=='uff':
          table.loc[i,'step_time']=start_test_onnx(table,i,[1])

table.to_csv('table.csv')









