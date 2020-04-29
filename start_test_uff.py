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
exec_root='/usr/src/TensorRT-7.0.0.11/TensorRT-7.0.0.11/bin/'
def start_test_uff(table, seq, batch):
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
    command = exec_root+'trtexec --uff={} --output={}{} --batch={}'.format(uff_name, output_nodes_name, input_params, batch)
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





batch_sizes=table['batch_size']
for i in range(len(table)):
#      if table.tasks[i].split('.')[1]=='pb':
#          table.loc[i,'step_time']=start_test_pb(table,i,batch_sizes[i])
      if table.tasks[i].split('.')[-1]=='uff':
          table.loc[i,'step_time']=start_test_uff(table,i,batch_sizes[i])
#      if table.tasks[i].split('.')[-1]=='onnx':
#          table.loc[i,'step_time']=start_test_onnx(table,i,batch_sizes[i])

table.to_csv('table.csv')










