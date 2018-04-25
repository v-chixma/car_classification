./dataset_namelist保存的是数据集的namelist

./demo_images 保存的是demo图片

./ImageNet_pretrained_model 预训练模型

./lib 保存的是训练所需的prototxt

./log 训练的日志

./results 训练的结果

./tools 一些脚本

cars_annos.mat 数据集的标签信息


训练和测试之前需要添加caffe的路径：/home/chixma/software/caffe/build/tools/caffe

训练：caffe train --solver="./lib/solver_finetune.prototxt" --weights="./ImageNet_pretrained_model/squeezenet_v1.1.caffemodel" --gpu=2

测试：caffe train --solver="./lib/solver_test.prototxt" --weights="./results/finetune_iter_15105.caffemodel" --gpu=2

运行demo之前需要添加pycaffe的路径sys.path.insert(0, r'/home/chixma/software/caffe/python/')

demo：python ./tools/demo.py ./demo_images/000066.jpg

解析日志，得到loss和accuracy曲线：python ./tools/parse_log.py ./log/finetune_221_train_val1_300x300.log

