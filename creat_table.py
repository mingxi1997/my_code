
import pandas as pd

with open('all_nodes.txt','r')as f:
    all_nodes=eval(f.read())
with open('assignment.txt','r')as f:
    tasks=f.read().split('\n')
tasks=[task for task in tasks if task!='']



output_node=[node[1][0] for node in all_nodes]

def wash(nas):
    nas=nas[0]
    result=[(na[0],na[1:])for na in nas]
    return result
input_nodes_shape=[wash(node) for node in all_nodes]

n_tasks=[task[:-3]+'_FP16.pb' for task in tasks]
n_tasks2=[task[:-3]+'.uff' for task in tasks]
n_tasks3=[task[:-3]+'.onnx' for task in tasks]

tasks.extend(n_tasks)
tasks.extend(n_tasks2)
tasks.extend(n_tasks3)
table=pd.DataFrame()
table['tasks']=tasks
table['output_node']=output_node*4
table['input_nodes']=input_nodes_shape*4
table['step_time']=0
table['batch_size']=1
table['precision']='fp16'
table.to_csv('table.csv')
