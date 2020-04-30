import pandas as pd
import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
import time
import subprocess
import  os
import re
table=pd.read_csv('table.csv')



logs_dir = './test/'
if not os.path.exists(logs_dir):
   os.makedirs(logs_dir)
trt_root='/usr/src/tensorrt/bin'




def start_test_pb(table,seq,batch):
  tf.reset_default_graph()
  run_metadata = tf.RunMetadata()
  input_pb=table.iloc[seq].tasks

  input_set=["import/"+input_node[0]+":0" for input_node in eval(table.iloc[seq].input_nodes)]
  input_size=[input_node[1]*batch for i,input_node in enumerate(eval(table.iloc[seq].input_nodes))]

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





def start_test_uff(table, seq, batch):
    tf.reset_default_graph()
    uff_name = table.iloc[seq].tasks
    input_nodes_str = table.iloc[seq].input_nodes
    output_nodes_name = table.iloc[seq].output_node
  
    input_nodes_list = eval(input_nodes_str)  # str=>list
    input_params = ""  # concat uffInput params
    for node in input_nodes_list:
            input_params += " --uffInput="+node[0] + ","
            node_size = node[1]
            node_size = node_size[1:]
            for size in node_size:
                input_params += size + ","
            input_params = input_params[:-1]  # remove redundent comma
    command = trt_root+'/trtexec --uff={} --output={}{} --batch={}'.format(uff_name, output_nodes_name, input_params, batch)
    print('*'*30)
    print(command)
    print('*'*30)
    log = subprocess.run(command, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
            # write log
    log_file = os.path.join(logs_dir, os.path.basename(uff_name).split('.')[0] + "_uff.txt")
    if os.path.isfile(log_file):
                os.remove(log_file)
    with open(log_file, 'w') as f:
                f.write(log)
    return -1

def get_means(log):
    result = re.search(r"mean: (\d+(\.)?\d+) ms", log)
    if result is None:
        return -1
    return result.group(1)




def start_test_onnx(table, seq, batch_size):
    tf.reset_default_graph()
    pb_name = table.iloc[seq].tasks
    input_nodes_str = table.iloc[seq].input_nodes
    input_nodes_names = eval(input_nodes_str)
    def multi_batch(strings,batch_size):
        strings[0]=str(int(strings[0])*batch_size)
        return strings
    shapes = "--shapes=" + ','.join([n[0] + ":0:" + 'x'.join(multi_batch(n[1],batch_size)) for n in input_nodes_names]).replace("-", "")
    
    onnx_name = os.path.basename(pb_name).split('.')[0] + ".onnx"
    trt_root='/usr/src/tensorrt/bin'
    command = trt_root+'/trtexec --onnx={} {} --fp16'.format(onnx_name, shapes)
  
#    print(command)


    log_onnxtest = subprocess.run(command, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
        # save means
    result = get_means(log_onnxtest)
        # write log
   # log_file = os.path.join(logs_dir, os.path.basename(pb_name).split('.')[0] + "_onnx.txt")\
    log_file = './test/'+pb_name[:-3]+'_onnx.txt'
    with open(log_file, 'w') as f:
            f.write(log_onnxtest)
    print(result)
    return result



batch_sizes=table['batch_size']
for i in range(len(table)):
      if table.tasks[i].split('.')[-1]=='pb':
          table.loc[i,'step_time']=start_test_pb(table,i,batch_sizes[i])
      if table.tasks[i].split('.')[-1]=='uff':
          table.loc[i,'step_time']=start_test_uff(table,i,batch_sizes[i])
      if table.tasks[i].split('.')[-1]=='onnx':
          table.loc[i,'step_time']=start_test_onnx(table,i,batch_sizes[i])

table.to_csv('table.csv')
