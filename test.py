import pandas as pd
import subprocess
import tensorflow as tf
import numpy as np
import time
import os

table = pd.read_csv('table.csv')
tasks = table.tasks.tolist()
tasks = [task for task in tasks if os.path.splitext(task)[1] == '.uff']


#print(tasks)
#batches = [1, 2, 5]
# ./trtexec --uff=e_trainingFalse.opt.uff --output=latents_out --uffInput=images_in,512,512,3     --uffNHWC（可选） --batch =8
logs_dir = './test/'
# create log directory
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# excute trtexec command by subprocess,return log information
def exec_uff(uff_name, input_nodes_str,batch):
    input_nodes_list = eval(input_nodes_str) # str=>list
    input_params='' # concat uffInput params
    for node in input_nodes_list:
    
        input_params=input_params+node[0]+","
        node_size = node[1]
        node_size=node_size[1:]
#        def reduce_head(nodes):
#            return  [node[1:] for node in nodes]
#       
        for  n in node_size:
            input_params+=n+','
      
        input_params = input_params[:-1] # remove redundent comma
        
      
        print('./trtexec --uff={} --output=latents_out --uffInput={} --uffNHWC --batch ={}'.format(uff_name,input_params,batch))
    return subprocess.run('./trtexec --uff={} --output=latents_out --uffInput={} --uffNHWC --batch ={}'.format(uff_name,input_params,batch), shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')

def start_test(table, seq, batches):
    uff_name = table.iloc[seq].tasks
    input_nodes_str = table.iloc[seq].input_nodes
    for batch in batches:
        log = exec_uff(uff_name, input_nodes_str, batch)
        # write log
        log_file = os.path.join(logs_dir, os.path.basename(uff_name).split('.')[0] + "_uff.txt")
        if os.path.isfile(log_file):
            os.remove(log_file)
        with open(log_file, 'w') as f:
            f.write(log)


batches=[1]
seq=8
start_test(table,seq,batches)
table.to_csv('table.csv')
