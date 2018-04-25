import sys 
# sys.path.append(r'../py-faster-rcnn/caffe-fast-rcnn/python/')
sys.path.insert(0, r'/home/chixma/software/caffe/python/')
import caffe
import numpy as np 
import cv2 
import time 
import argparse
import matplotlib.pyplot as plt 
import scipy.io as scio
import os
matfile = scio.loadmat(r'./cars_annos.mat')

def get_gtLabel(imName):
	annos = matfile['annotations'][0]
	for ann in annos[:]:
		tmpName = os.path.basename(ann[0][0])
		if tmpName == imName:
			return ann[-2][0,0]-1
	return -1

def get_className(classIdx):
	classNames = matfile['class_names'][0]
	names = []
	for className in classNames:
		# print className[0]
		names.append(className[0])
	return names[classIdx]

def load_image_for_pretrained_caffemodel(imPath):
	im = cv2.imread(imPath)
	if len(im.shape) == 2:
		new_im = np.zeros((im.shape[0],im.shape[1],3))
		new_im[:,:,0] = im
		new_im[:,:,1] = im 
		new_im[:,:,2] = im 
		im = new_im 
	im = cv2.resize(im,(227,227),interpolation = cv2.INTER_LINEAR)
	mean = np.array([104,117,123])
	im = im - mean
	im = im.transpose((2,0,1))#HWC -> CHW
	return im
	
def infer(imPath):
	
	caffe.set_device(0)
	caffe.set_mode_gpu()
	deploy_prototxt = r'./lib/deploy.prototxt'
	caffe_model = r'./results/finetune_iter_10250.caffemodel'
	net = caffe.Net(deploy_prototxt,caffe_model,caffe.TEST)
	testIm = load_image_for_pretrained_caffemodel(imPath)
	net.blobs['data'].reshape(1,3,227,227)
	net.blobs['data'].data[0,:,:,:] = testIm
	net.forward()
	testIm_feature = net.blobs['prob'].data.copy()
	testIm_feature = testIm_feature.squeeze()
	# print testIm_feature.shape
	classIdx = np.argmax(testIm_feature)
	return classIdx


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('demo_image',type=str,help='the demo image to be test') #,default = './000066.jpg'
	args = parser.parse_args()
	# print args 
	return args

if __name__ == '__main__':
	args = get_args()
	imPath = args.demo_image
	start = time.time()
	classIdx = infer(imPath)
	total_eval_time = time.time()-start 
	print 'Inference class Index:', classIdx
	className = get_className(classIdx)
	print 'Inference class name:', className
	print 'Inference Time for {}: {}s'.format(imPath, total_eval_time)
	gt_classIdx = get_gtLabel(os.path.basename(imPath))
	if gt_classIdx != -1:
		print 'Inference result is: ',gt_classIdx == classIdx
		if gt_classIdx != classIdx:
			print 'Ground truth class name:', get_className(gt_classIdx)
	else:
		print 'No gt label for this image!'


	
	
					

	





