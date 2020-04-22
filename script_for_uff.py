import subprocess


#get tasks 
with open('assignment.txt','r')as f:
    tasks=f.read().split('\n')
tasks=[task for task in tasks if task!='']

#convert to uff and save log

logs=[subprocess.run('convert-to-uff '+ task,shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8') for task in tasks]

#write log
for i,log in enumerate(logs):
    with open(tasks[i][:-3]+'_log'+'.txt','w')as f:
        f.write(log)

#save input and out put
for log in logs:
    nodes=log.split('=== Automatically deduced output nodes ===')
    innodes=[]
    outputnodes=[]
    for row in nodes[0].split('\n'):
        if 'name:' in row:
             innodes.append(row.split(':')[1].replace('"','').replace(' ',''))
    for row in nodes[1].split('\n'):
        if 'name:' in row:
             outputnodes.append(row.split(':')[1].replace('"','').replace(' ',''))
    print(innodes,outputnodes)
