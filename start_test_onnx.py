import pandas as pd
import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
import time
import subprocess
import  os
import re
table=pd.read_csv('table.csv')
run_metadata = tf.RunMetadata()



logs_dir = './test/'
if not os.path.exists(logs_dir):
   os.makedirs(logs_dir)


def get_means(log):
    result = re.search(r"mean: (\d+(\.)?\d+) ms", log)
    if result is None:
        return -1
    return result.group(1)




def start_test_onnx(table, seq, batch_size,precision):
    pb_name = table.iloc[seq].tasks
    input_nodes_str = table.iloc[seq].input_nodes
    input_nodes_names = eval(input_nodes_str)
    def multi_batch(strings,batch_size):
        strings[0]=str(int(strings[0])*batch_size)
        return strings
    shapes = "--shapes=" + ','.join([n[0] + ":0:" + 'x'.join(multi_batch(n[1],batch_size)) for n in input_nodes_names]).replace("-", "")
    
    onnx_name = os.path.basename(pb_name).split('.')[0] + ".onnx"
    command = '/usr/src/tensorrt/bin/trtexec --onnx={} {} --{}'.format(onnx_name, shapes,precision)
  
#    print(command)


    log_onnxtest = subprocess.run(command, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
        # save means
    result = get_means(log_onnxtest)
        # write log
   # log_file = os.path.join(logs_dir, os.path.basename(pb_name).split('.')[0] + "_onnx.txt")\
    log_file = './test/'+pb_name[:-3]+'_onnx.txt'
    with open(log_file, 'w') as f:
            f.write(log_onnxtest)
    return result
batch_sizes=table['batch_size']
for i in range(len(table)):
#      if table.tasks[i].split('.')[1]=='pb':
#          table.loc[i,'step_time']=start_test_pb(table,i,batch_sizes[i])
#      if table.tasks[i].split('.')[1]=='uff':
#          table.loc[i,'step_time']=start_test_uff(table,i,[1])
      if table.tasks[i].split('.')[-1]=='onnx':
          table.loc[i,'step_time']=start_test_onnx(table,i,batch_sizes[i],table.iloc[i]['precision'])

table.to_csv('table.csv')
