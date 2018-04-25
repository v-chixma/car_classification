#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-01-29 21:13:35
# @Author  : chixma (${email})
# @Link    : ${link}
# @Version : $Id$
import cv2
import os
SIZE = 300
dataRootPath = r'/disk1/chixma/dataset/car/train_test/images'
# savePath = r'/disk1/chixma/dataset/car/train_test/227x227images'
savePath = r'/disk1/chixma/dataset/car/train_test/300x300images'
if not os.path.isdir(savePath):
	os.mkdir(savePath)

if __name__ == '__main__':
	count = 0
	for rootDir, subDirs, filenames in os.walk(dataRootPath):
		if len(filenames) == 0:
			continue

		for index, filename in enumerate(filenames):
			im = cv2.imread(os.path.join(rootDir,filename))
			if len(im.shape) == 2:
				new_im = np.zeros((im.shape[0],im.shape[1],3))
				new_im[:,:,0] = im
				new_im[:,:,1] = im
				new_im[:,:,2] = im
				im = new_im
			aspect_ratio = im.shape[1]*1.0/im.shape[0]
			im = cv2.resize(im,(SIZE,SIZE),interpolation = cv2.INTER_LINEAR)
			# im = cv2.resize(im,(227,227),interpolation = cv2.INTER_LINEAR)
			cv2.imwrite(os.path.join(savePath,filename),im)
			count += 1
			print count 





