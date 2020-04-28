# encoding=utf-8
import subprocess
import re
import os


def get_name_size(data):
    input_list = []
    output_list = []
    result = re.search(r"=== Automatically deduced input nodes ===.*?===", data, re.S)
 
    if result:
        input_data = result.group(0)  
      
        names = re.findall(r"name.*?\,|name.*?\]", input_data, re.S)
       
        if len(names) > 0:
            for name in names:
                
                info = re.findall(r"name.*|size.*", name, re.M)
                if len(info) > 0:
                    name_list = []
                  
                    for val in info:
                        pos = val.find(":")
                        if pos >= 0:
                            name_list.append(val[pos + 1:].replace('"', '').strip(" "))
                    input_list.append(name_list)

  
    result = re.search(r"=== Automatically deduced output nodes ===.*?===", data, re.S)

    if result:
        output_data = result.group(0)  
       
        names = re.findall(r"name.*?\,|name.*?\]", output_data, re.S)
      
        if len(names) > 0:
            for name in names:
           
                info = re.findall(r"name.*", name, re.M)
                if len(info) > 0:
                   
                    for val in info:
                        pos = val.find(":")
                        if pos >= 0:
                            output_list.append(val[pos + 1:].replace('"', '').strip(" "))

    return input_list, output_list



#get tasks 
with open('assignment.txt','r')as f:
    tasks=f.read().split('\n')
tasks=[task for task in tasks if task!='']

#convert to uff and save log

logs=[subprocess.run('convert-to-uff '+ task,shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8') for task in tasks]
logs_dir = './transform/'
# create log directory
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
# get nodes and save
all_nodes = []
for i,log in enumerate(logs):
    input_list, output_list=get_name_size(log)
    all_nodes.append((input_list, output_list))
    # write log
    log_file = os.path.join(logs_dir,os.path.basename(tasks[i]).split('.')[0]+".txt")
    if os.path.isfile(log_file):
        os.remove(log_file)
    with open(log_file, 'w') as f:
        f.write(log)


with open('all_nodes.txt','w')as f:
    f.write(str(all_nodes))


def get_num(text):
    return [int(t) for t in text]

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.platform import gfile




for i,node in enumerate(all_nodes):
    tf.reset_default_graph()
    inputs,outputs=node
    outputs_name=outputs[0]
    inputs_detail=[(input[0],get_num(input[1:])) for input in inputs]

    input_pb=tasks[i]
    print(input_pb)
    input_node_set=["import/"+input_detail[0]+":0" for input_detail in inputs_detail]

    output_node=outputs[0]

    precision="FP16"
    with tf.Session() as sess:

        with tf.gfile.GFile(input_pb,'rb') as f:
            frozen_graph = tf.GraphDef()
            frozen_graph.ParseFromString(f.read())
        # Now you can create a TensorRT inference graph from your
        # frozen graph:
        converter = trt.TrtGraphConverter(

                    input_graph_def=frozen_graph,
                    minimum_segment_size=3,
                    precision_mode=precision,  #INT8/FP16/FP32
                    nodes_blacklist=[output_node]) #output nodes
        trt_graph = converter.convert()
        #save model

    with gfile.FastGFile(tasks[i][:-3]+"_FP16.pb",'wb') as f:
            f.write(trt_graph.SerializeToString())
