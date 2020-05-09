#encoding=utf-8
import pandas as pd
import pandas as pd
import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
import time
import subprocess
import os
import re
import time
import LogManager


class TableManager(object):
    def __init__(self):
        self.batches = []
        self.tasks = []
        self.precisions = []
        self.input_nodes_shape = []
        self.output_node = []
        self.table_name = "./table.csv"
        self.__normal_precision = ['fp16', 'fp32', 'int8']

    def ParseFromAssignment(self, filename):
        table = pd.read_csv(filename)
        self.tasks = table.tasks.values.tolist()
        for i in range(len(table)):
            batches_list = []
            str_batches = str(table.iloc[i].batch_size)
            batches = str_batches.split(',')
            for batch in batches:
                batches_list.append(batch.strip(' '))
            if len(batches_list) == 0:
                batches_list.append('1')
            self.batches.append(batches_list)

            precision_list = []
            str_precisions = table.iloc[i].precision
            precisions = str_precisions.split(',')
            for precision in precisions:
                precision_lower = precision.lower().strip(' ')
                if precision_lower in self.__normal_precision:
                    precision_list.append(precision_lower)
            if len(precision_list) == 0:
                precision_list.append('fp16')
            self.precisions.append(precision_list)

    def get_all_nodes(self):
        with open('all_nodes.txt', 'r')as f:
            all_nodes = eval(f.read())
        self.output_node = [node[1][0] for node in all_nodes]

        def wash(nas):
            nas = nas[0]
            result = [(na[0], na[1:]) for na in nas]
            return result

        self.input_nodes_shape = [wash(node) for node in all_nodes]

    def create_table(self):
        table = pd.DataFrame()
        table['tasks'] = []
        table['output_node'] = []
        table['input_nodes'] = []
        table['step_time'] = 0
        table['batch_size'] = 1
        table['precision'] = 'fp16'
        for index in range(len(self.tasks)):
               table.loc[len(table)] = [self.tasks[index], self.output_node[index], self.input_nodes_shape[index],'0', '1', 'fp32']
               for precision in self.precisions[index]:
                   if precision=='fp16':
                       table.loc[len(table)] = [self.tasks[index][:-3]+"_"+precision+".pb", self.output_node[index], self.input_nodes_shape[index],'0', '1', precision]
        for index in range(len(self.tasks)):
            for precision in self.precisions[index]:
                for batch in self.batches[index]:
                    table.loc[len(table)] = [self.tasks[index][:-3]+".uff", self.output_node[index], self.input_nodes_shape[index],
                                             '0', batch, precision]
        for index in range(len(self.tasks)):
            for precision in self.precisions[index]:
                for batch in self.batches[index]:
                    table.loc[len(table)] = [self.tasks[index][:-3]+".onnx", self.output_node[index], self.input_nodes_shape[index],
                                             '0', batch, precision]
        table.to_csv(self.table_name, index=False)


class TestManager(object):
    def __init__(self):
        self.exec_root = '/usr/src/tensorrt/bin/'
        self.table=pd.read_csv('table.csv')   
    def start_test_pb(self, table, seq):
        tf.reset_default_graph()
        run_metadata = tf.RunMetadata()
        input_pb =table.iloc[seq].tasks
        batch = table.iloc[seq].batch_size
        input_set = ["import/" + input_node[0] + ":0" for input_node in eval(table.iloc[seq].input_nodes)]
        input_size = [input_node[1] * batch for i, input_node in enumerate(eval(table.iloc[seq].input_nodes))]
        log_writer = LogManager.LogManager('./json/')
        output_node = table.iloc[seq].output_node

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

            with tf.gfile.GFile(input_pb, 'rb') as f:
                frozen_graph = tf.GraphDef()
                frozen_graph.ParseFromString(f.read())

            tf.import_graph_def(
                frozen_graph,
                return_elements=[output_node])
            output_node = "import/" + output_node + ":0"

            for step in range(251):

                if step == 150:
                    start_time = time.time()

                #        precision_for_np=np.float16 if precision=="FP16" else np.float32
                precision_for_np = np.float32

                def wash(shape):
                    return [abs(int(s)) for s in shape]

                input_shapes = [wash(shape) for shape in input_size]
                data_shapes = [np.random.randn(*wash(shape)).astype(precision_for_np) for shape in input_shapes]

                feed_dict = {input_set[i]: data_shapes[i] for i in range(len(input_set))}

                sess.run(output_node, feed_dict=feed_dict)

                if step == 250:
                    sess.run(output_node, feed_dict=feed_dict,
                             options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                             run_metadata=tf.RunMetadata())
                    trace = timeline.Timeline(step_stats=tf.RunMetadata().step_stats)
                    log_writer.WriteText(trace.generate_chrome_trace_format(), input_pb[:-3] + '.json')
                if step == 249:
                    end_time = time.time()
        return (end_time - start_time) / 100

    def test_pb(self):
       # table = pd.read_csv('table.csv')
        for i in range(len(self.table)):
            if self.table.tasks[i].split('.')[-1] == 'pb':
                print('*'*30 +"starting test {} ...".format(self.table.tasks[i]))
                try:
                  result = self.start_test_pb(self.table, i)
                except Exception as e:
                  print("*"*30+"error info:{}".format(e))
                  continue
                self.table.loc[i,'step_time']=result
                print('*' * 30)
                print(self.table.tasks[i])
                print(result)
     #   table.to_csv('table.csv', index=False)

    def get_means(self, log):
        result = re.search(r"mean: (\d+(\.)?\d+) ms", log)
        if result is None:
            return -1
        return result.group(1)

    def start_test_onnx(self, table, seq):
        onnx_name = table.iloc[seq].tasks
        batch_size = table.iloc[seq].batch_size
        precision = table.iloc[seq].precision
        input_nodes_str = table.iloc[seq].input_nodes
        input_nodes_names = eval(input_nodes_str)
        def multi_batch(strings,batch_size):
              strings[0]=str(int(strings[0])*batch_size)
              return strings
        shapes = "--shapes=" + ','.join([n[0] + ":0:" + 'x'.join(multi_batch(n[1],batch_size)) for n in input_nodes_names]).replace("-", "")

        # onnx_name = os.path.basename(onnx_name).split('.')[0] + ".onnx"
        command = '/usr/src/tensorrt/bin/trtexec --onnx={} {} --{}'.format(onnx_name, shapes,precision)
        # command = self.exec_root + 'trtexec --onnx={} {} --{}'.format(onnx_name, shapes, precision)

        print('*' * 30)
        print(command)
        print('*' * 30)

        log_onnxtest = subprocess.run(command, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
        # save means
        result = self.get_means(log_onnxtest)
        # write log
        log_writer = LogManager.LogManager('./logs/')
        log_writer.WriteText(log_onnxtest, onnx_name[:-5] + '.txt')
        return result
    def test_onnx(self):
   #     table = pd.read_csv('table.csv')
        for i in range(len(self.table)):
            if self.table.tasks[i].split('.')[-1] == 'onnx':
                print('*'*30 +"starting test {} ...".format(self.table.tasks[i]))
                str_result=self.start_test_onnx(self.table, i)
                float_reuslt =str(float(str_result)/1000.0)
                self.table.loc[i, 'step_time'] = float_reuslt

   #     table.to_csv('table.csv',index=False)

    def start_test_uff(self, table, seq):
        uff_name = table.iloc[seq].tasks
        batch = table.iloc[seq].batch_size
        precision = table.iloc[seq].precision
        input_nodes_str = table.iloc[seq].input_nodes
        output_nodes_name = table.iloc[seq].output_node

        input_nodes_list = eval(input_nodes_str)  # str=>list
        input_params = ""  # concat uffInput params
        for node in input_nodes_list:
            input_params += " --uffInput=" + node[0] + ","
            node_size = node[1]
            node_size = node_size[1:]
            for size in node_size:
                input_params += size + ","
            input_params = input_params[:-1]  # remove redundent comma
        command = self.exec_root + 'trtexec --uff={} --output={}{} --batch={} --{} --uffNHWC'.format(uff_name, output_nodes_name,
                                                                                 input_params, batch,precision)
        print('*' * 30)
        print(command)
        print('*' * 30)
        log = subprocess.run(command, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
        # write log
        log_writer = LogManager.LogManager('./logs/')
        log_writer.WriteText(log,uff_name[:-4] + ".txt")
        result = self.get_means(log)
        return result
    def test_uff(self):
   #     table = pd.read_csv('table.csv')
        for i in range(len(self.table)):
            if self.table.tasks[i].split('.')[-1] == 'uff':
                print('*'*30 +"starting test {} ...".format(self.table.tasks[i]))
                str_result=self.start_test_uff(self.table, i)
                float_reuslt =str(float(str_result)/1000.0)
                self.table.loc[i, 'step_time'] = float_reuslt
    #    table.to_csv('table.csv',index=False)
    def save_table(self):
        self.table.to_csv('table.csv',index=False)
if __name__ == "__main__":

    test = TestManager()
    print('*'*30 +"starting test onnx ...")
    test.test_onnx()
    print('*'*30 +"starting test uff ...")
    test.test_uff()
    print('*'*30 +"starting test pb ...")
    test.test_pb()
    print('*'*30 +"starting save table ...")
    test.save_table()
