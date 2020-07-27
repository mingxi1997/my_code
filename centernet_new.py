I fixed it.
cuda=10.0, python=3.6, anaconda=Anaconda3-5.3.1-Linux-x86_64.sh
1- git clone https://github.com/xingyizhou/CenterNet.git
2- conda create --name CenterNet python=3.6
3- conda activate CenterNet
4- pip install -r requirements.txt
5- conda install pytorch=1.1 torchvision(the pip channel use https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free)
6- cd CenterNet\src\lib\external
python setup.py build_ext --inplace
7- cd /CenterNet/src/lib/models/networks
rm -r DCNv2
git clone https://github.com/CharlesShang/DCNv2.git
cd /CenterNet/src/lib/models/networks/DCNv2
python setup.py build develop
8- down ctdet_coco_dla_2x.pth from this https://drive.google.com/file/d/1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT/view and put it in /Centernet/models
9- cd CenterNet/src/, and run:
python demo.py ctdet --demo ../images/24274813513_0cfd2ce6d0_k.jpg --load_model ../models/ctdet_coco_dla_2x.pth
10- done!
