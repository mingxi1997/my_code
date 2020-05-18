# my_code
1	使用说明
1.1	快速使用说明
STEP1：环境准备
1、需要TF1.5、onnx、TensorRT以及Cuda、Cudnn等相关软件；推荐使用容器，当前已将依赖打好，如21环境上inf_gpu_tool。其他设备部署，可参考下文中的容器迁移方法
	2、下载脚本
	https://gitlab.huawei.com/c00506053/dnmetis/tree/master/gpu_perf_tool
    
3、拉起容器：
进入步骤2中的脚本路径：执行下面的lauch.sh即可


STEP2：准备模型和用例
1、将待测PB模型放到model路径下，并填写待测用例
Workspace/gpu_perf_tool/model放模型
2、填写你要测的模型名、batch和精度到assignment.csv
Workspace/gpu_perf_tool下填写assignment.csv
 
Tasks：填写待测的模型
batch_size/precision：填写要测试的batch大小和精度大小，如果多个值，用逗号隔开即可

STEP3：自动执行
方式一：一键执行：执行gpu_perf_tool下的start.sh
方式二：分步执行
1）	先执行python transform.py  生成用例和模型。 
可查看table.csv 中自动生成的用例是否符合预期
可查看modle下转换的onnx/UFF/TF-TRT模型
2）	执行python start_test.py执行性能测试，并自动汇总结果
生成随机值数据，对模型进行性能测试，当前会自动进行的测试方法包括
a.	Tf 原生模型sess.run 执行 
b.	Tf-trt优化模型（FP16）方式 sess.run执行
C.  TensorRT 利用trtexec对原始模型转为onnx执行
d.  TensorRT 利用trtexec对原始模型转为uff执行
  PS：转换模型日志可参考create_log目录， 执行日志可参考logs目录

STEP4：结果查看
	当前自动生成性能时延数据和time-line打点 
推理性能： gpu_perf_tool下的table.csv，数据是一轮迭代E2E的平均latency（单位step/ms）。如需吞吐和自己换算 throughput = batchsize*1000/ latency （单位 images/s）
 
     Time-line打点 ：gpu_perf_tool/json下查看，当前对sess_run方式会每个模型在性能统计后再跑一轮time-line数据
	
补充说明：
1） 部分模型受限TRT支持的算子，可能无法转换或执行成功，如最终某case的结果为0或-1，说明执行失败。可检测日志详细信息
2） 当前只支持单输出节点
3） 关于动态shape的模型，因目前是自动检索输入node及shape，动态时无法自动获取输入大小，规避方法查看FAQ中的“动态shape”，建议测试前修改pb模型

1.2	目录文件说明
1.	|-- all_nodes.txt                      #过程文件：自动获取的输入输出节点
2.	|-- assignment.csv  			  #用户输入：填写要测试的pb模型
3.	|-- create_log                    #转换模型的日志文件夹
4.	|   |-- create_model.log              #LOG：step1中转换模型的全量日志
5.	|   |-- esrgan_uff.txt                #LOG:step1中转换UFF的日志
6.	|-- json                           #time-line日志
7.	|   |--resnet18_1.json                #输出：tf方式测试自动保存的time-line
8.	|-- LogManager.py                      #日志获取实现
9.	|-- logs  				       #执行结果日志
10.	|   |-- resnet18_1_INT8_onnx.log     #trtexec对每个具体case执行日志
11.	|   |-- test.log                       #单次任务全量日志
12.	|-- model  
13.	|   |-- resnet50_v1.pb                 #将测试模型放在该路径下
14.	|-- start.sh                       #一次执行生成用例->模型转换->推理执行->数据收集
15.	|-- start_test.py                  #推理执行实现
16.	|-- table.csv                      #性能结果汇总
17.	|-- tableManager.py                #表格操作文件
18.	|-- tensorflow_util.py            
19.	`-- transform.py                    #模型转换实现
蓝色为用户需要关注的输入输出的部分

1.3	参数配置说明

Transform.py中的可配参数：
	"--create_log_path", default='./create_log/    模型转换保存日志
	"--model_path", default='./model/'		   模型获取地址
	"--renew_model", default='false'			   当已存在转换后的模型时，是否重新生成

Start_test.py中的可配参数
"--model_path", default='./model/',      模型获取地址
"--input_precision", default='FP32'	   灌入数据的精度，默认FP32
"--step_num", default=120			   在线执行跑的step数，性能数据会取最后20个迭代
	

2	实现内容
 
2.1	TF离线执行
 
TensorRT是无法直接解析TF的PB模型，所以PB模型当前在TensorRT上离线部署推理大概会有这么两条路
       TF(PB) --> 转UFF （N提供转换工具） -> TensorRT推理
            --> 转ONNX （ONNX有工具，非N提供） -> TensorRT推理
前者转UFF，TensorRT提供工具，但是支持的TF算子比较有限，网络中有不支持的就转不了； 后者因为TensorRT支持解析onnx模型，所以我们可以用ONNX提供的开源工具TF2ONNX，把pb转onnx再拿到TensorRT上执行。

测试采用N提供的trtexec工具;
https://github.com/NVIDIA/TensorRT/tree/master/samples/opensource/trtexec

2.1.1	Trtexec执行UFF
模型转换：
通过nvidia 的 convert-to-uff  xxx.pb实现

执行命令：
./trtexec --uff=resnet50.uff --output=ArgMax --uffInput=input_tensor,224,224,3 --batch=32  --uffNHWC  

数据获取：
当前以E2E的性能作为采集指标，同NPU的整网测试一致
 

2.1.1	Trtexec执行ONNX
转换命令
通过onnx开源的tf2onnx实现，如
python3 -m  tf2onnx.convert  --input  resnet-101-tf.pb  --output resnet101.onnx --inputs input:0  --outputs resnet_v1_101/predictions/Softmax:0

执行命令参考：
./trtexec --onnx=resnet50.onnx   --shapes=\'input_tensor:0\':8x224x224x3    --optShapes=\'input_tensor:0\':8x224x224x3

性能获取同UFF

2.2	TF在线执行
2.2.1	TF原生执行
1、Load原生pb 生成随机数据通过sess_run运行，当前默认输入数据为FP32，提供可改参数
2、当前性能数据获取为执行100个step待稳定执行后，再获取20个step做平均。执行step数量提供参考可调

2.2.2	TF-TRT执行
TRT7提供的集成在Tensorflow中的在线优化，通用性较好，如有不支持的算子会原生框架执行。故可作为因TensorRT不支持算子而无法执行的pb的补充验证
转换方法如下，通过如上命令对原pb进行优化，如想了解前期实验可见DBOX路径  ：  海思/Tuscany/Docs/V100R001/V100R001C10/02. Project Folder/03. I&V Folder/16. 性能测试/benchmark/5_测试资产/总结文档/TF-TRT实践--周啸.docx
1.	    converter = trt.TrtGraphConverter(  
2.	            input_graph_def=frozen_graph,  
3.	            minimum_segment_size=3,  
4.	            precision_mode="FP16",  #INT8/FP16/FP32
5.	            nodes_blacklist=["final_masks"]) #output nodes  
转换后的pb，同tf原生模型方式执行




2.3	Time-line采集
方法：同周莉同学总结的sess run方法
http://3ms.huawei.com/hi/group/3554771/thread_8121676.html?mapId=9927922&for_statistic_from=all_group_forum

	采集：
 	在sess_run性能统计后单独执行一轮做trace，不影响性能时延结果

3	常见问题和说明
3.1	容器迁移及修改

1、容器拉起
nvidia-docker run -it --rm --ipc=host --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v /home/:/workspace/ inf_gpu_tool:v1.0 bash
	
/home/model :/workspace/data  为本地路径和映射到容器内路径，根据实际情况修改


2、容器迁移
   保存容器docker save nvcr.io/nvidia/pytorch >  xxxxx.ta      //xxx为你起的名字，随意 
   将 xxxxx.ta拷贝到新环境上
   docker load < xxxxx.ta
   
容器安装可参考利霞之前小结的文档，DBOX：
海思/Tuscany/Docs/V100R001/V100R001C10/02. Project Folder/03. I&V Folder/16. 性能测试/benchmark/5_测试资产/总结文档/竞品DGX-2环境Resnet50网络训练-魏利霞 .docx

3、容器修改
 1）先拉起容器， 在容器中安装软件或修改
2）另外一个用户在容器外通过docker ps查看你运行的容器id ，例如为3bd0eef03413。 
3） docker commit 3bd0eef03413  demo：v1.3     // 3bd0eef03413 为容器id  ， demo为新镜像名词， v1.3为tag 


3.2	涉及的工具链接
Nvidia NGC容器及内部软件：
https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow
https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/index.html

TF2ONNX转换工具：
https://github.com/onnx/tensorflow-onnx
命令参考

TensorRT相关链接：
https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html
TF-TRT相关链接：
https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html

Trtexec：
https://github.com/NVIDIA/TensorRT/tree/master/samples/opensource/trtexec


3.3	动态shape规避方法
现象：解析出来的HW为-1
如动态shape
 
 



或发现在转换后生成的用例table.csv中没shape信息
  
解决方法：
1、修改模型
   工具目录下有如下工具脚本，过程注意对原pb先备份
1）	利用python tensorflow_util.py –t 将pb模型转换为pbtxt
2）	文本打开pbtxt，修改其中placehold中的HWC dim为固定shape
3）	利用python tensorflow_util.py –p 将修改后pbtxt模型转换为pb
例如
node {
  name: "input"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
        dim {
          size: 299
        }
        dim {
          size: 299
        }
        dim {
          size: 3
        }
      }
    }
  }
}


2 修改table中的shape，将HWC写为固定值。调用python start_test.py执行
  第一方式相对稳妥
第二种方法测试单个模型目前实验只能适用部分TF在线方式和UFF方式



3.4	Time-line查看方法
1、	chrome流量器中运行chrome://tracing/ ，然后load保存出的json即可
2、	用蓓蓓提供的脚本可将json转换为xls（忽略里面ckpt的meta）
 

3.5	模型batch写死只能运行单batch
可能现象：在线运行方式只有某个batch可以运行，其他batch不行；或所有batch性能数据一样
 
建议同动态shape方法中方法，将原pb中的batch修改为动态

4	归档位置和后续计划
归档到TSE组的git路径
https://gitlab.huawei.com/c00506053/dnmetis

后续计划：
1、	补充其他框架的tensorRT方式执行：CAFFE\ONNX\PYTORCH
2、	Nvidia最新的DLPROF方式打点性能采样
3、	动态shape模型自动修改

