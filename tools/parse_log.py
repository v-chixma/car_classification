import matplotlib.pyplot as plt 
import string
import argparse

def parse(logFile):
	f = open(logFile,'r')
	lines = f.readlines()
	trainLoss = []
	valLoss =[]
	top1_accuracy = []
	top5_accuracy = []
	for line in lines:
		line = line.strip().split(' ')
		if len(line) > 12 and line[-1] == 'loss)' and line[-12] == 'Train':
			trainLoss.append(string.atof(line[-2]))
		elif len(line) > 12 and line[-1] == 'loss)' and line[-12] == 'Test':
			valLoss.append(string.atof(line[-2]))
		elif len(line) > 7 and line[-3] == 'accuracy':
			top1_accuracy.append(string.atof(line[-1]))
		elif len(line) > 7 and line[-3] == 'accuracy_top5':
			top5_accuracy.append(string.atof(line[-1]))
	plt.figure()
	plt.plot(trainLoss[5:])
	plt.legend(['train loss'],loc = 'upper right')
	plt.xlabel('iters')
	plt.ylabel('trainLoss')
	plt.grid('on')
	plt.figure()
	plt.plot(valLoss)
	plt.plot(top1_accuracy)
	plt.plot(top5_accuracy)
	plt.legend(['val loss','top1_accuracy','top5_accuracy'],loc = 'upper right')
	plt.xlabel('epoch')
	plt.ylabel('valLoss, top1_accuracy, top5_accuracy')
	plt.grid('on')
	plt.show()
def parse_args():
	parser = argparse.ArgumentParser(description='parse args')
	parser.add_argument('logFile',help='Path/to/log/file') #default='./log/finetune_221_trainval.log', 
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	# logFile = r'./log/finetune_221_trainval.log'
	args = parse_args()
	parse(args.logFile)