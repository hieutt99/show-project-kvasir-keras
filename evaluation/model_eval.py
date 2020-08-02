from evaluation.metrics import *
import cv2
import matplotlib.pyplot as plt
import numpy as np

def model_evaluate(model, train_generator, val_generator, test_generator, \
	batch_size = 8, train_size = 700, val_size = 100, test_size = 200, pass_train = False):
	'''
	run end evaluate generators train - val - test 
	evaluate F1 and Mean IOU on each set
	'''
	
	print('=========================================================')
	print('======================EVALUATION=========================')
	print('=========================================================')
	
	metrics = model.metrics_names
	print(metrics)

	if not pass_train:
		print('\n-------------On Training Set------------------------\n')
		res = model.evaluate(train_generator, steps = np.floor(train_size/batch_size), verbose = 1)
		print('________________________________')
		for i, metric in enumerate(metrics):
			print('%-20s: |   %.4f' %(metric, res[i]))
		# print('%-20s: |   %.4f' %('F1 Score', f1(res[7],res[8],res[9])))
		# print('%-20s: |   %.4f' %('IOU', mean_iou(res[7],res[8],res[9])))
		
		## dspvs
		print('%-20s: |   %.4f' %('F1 Score', f1(res[10],res[11],res[12])))
		print('%-20s: |   %.4f' %('IOU', mean_iou(res[10],res[11],res[12])))

		
		print('%-20s: |   %.4f' %('F1 Score out_2', f1(res[19],res[20],res[21])))
		print('%-20s: |   %.4f' %('IOU out_2', mean_iou(res[19],res[20],res[21])))

		print('%-20s: |   %.4f' %('F1 Score out_1', f1(res[28],res[29],res[30])))
		print('%-20s: |   %.4f' %('IOU out_1', mean_iou(res[28],res[29],res[30])))


	print('\n-------------On Validation Set----------------------\n')
	res = model.evaluate(val_generator, steps = np.floor(val_size/batch_size), verbose = 1)
	print('________________________')
	for i, metric in enumerate(metrics):
		print('%-20s: |   %.4f' %(metric,res[i]))
	# print('%-20s: |   %.4f' %('F1 Score', f1(res[7],res[8],res[9])))
	# print('%-20s: |   %.4f' %('IOU', mean_iou(res[7],res[8],res[9])))
	print('%-20s: |   %.4f' %('F1 Score', f1(res[10],res[11],res[12])))
	print('%-20s: |   %.4f' %('IOU', mean_iou(res[10],res[11],res[12])))

	
	print('%-20s: |   %.4f' %('F1 Score out_2', f1(res[19],res[20],res[21])))
	print('%-20s: |   %.4f' %('IOU out_2', mean_iou(res[19],res[20],res[21])))

	print('%-20s: |   %.4f' %('F1 Score out_1', f1(res[28],res[29],res[30])))
	print('%-20s: |   %.4f' %('IOU out_1', mean_iou(res[28],res[29],res[30])))


	print('\n-------------On Test  Set--------------------------\n')
	res = model.evaluate(test_generator, steps = np.floor(test_size/batch_size), verbose = 1)
	print('________________________')
	for i, metric in enumerate(metrics):
		print('%-20s: |   %.4f' %(metric,res[i]))
	# print('%-20s: |   %.4f' %('F1 Score', f1(res[7],res[8],res[9])))
	# print('%-20s: |   %.4f' %('IOU', mean_iou(res[7],res[8],res[9])))
	print('%-20s: |   %.4f' %('F1 Score', f1(res[10],res[11],res[12])))
	print('%-20s: |   %.4f' %('IOU', mean_iou(res[10],res[11],res[12])))

	
	print('%-20s: |   %.4f' %('F1 Score out_2', f1(res[19],res[20],res[21])))
	print('%-20s: |   %.4f' %('IOU out_2', mean_iou(res[19],res[20],res[21])))

	print('%-20s: |   %.4f' %('F1 Score out_1', f1(res[28],res[29],res[30])))
	print('%-20s: |   %.4f' %('IOU out_1', mean_iou(res[28],res[29],res[30])))


def testSetReveal(model, test_generator):
	print('=========================================================')
	print('======================testSetReveal======================')
	print('=========================================================')

	for i, batch in enumerate(test_generator):
		if i == 1:
			test_x, test_y = batch[0], batch[1]
			print(test_x.shape)
			print(test_y.shape)
			break

	metrics = model.metrics_names
	print(metrics)
	print('\n-------------On Test  Set--------------------------\n')
	res = model.evaluate(test_x,test_y)
	print('________________________')
	for i, metric in enumerate(metrics):
		print('%-20s: |   %.4f' %(metric,res[i]))
	# print('%-20s: |   %.4f' %('F1 Score', f1(res[7],res[8],res[9])))
	# print('%-20s: |   %.4f' %('IOU', mean_iou(res[7],res[8],res[9])))
	print('%-20s: |   %.4f' %('F1 Score X_output', f1(res[10],res[11],res[12])))
	print('%-20s: |   %.4f' %('IOU X_output', mean_iou(res[10],res[11],res[12])))

	print('%-20s: |   %.4f' %('F1 Score out_2', f1(res[19],res[20],res[21])))
	print('%-20s: |   %.4f' %('IOU out_2', mean_iou(res[19],res[20],res[21])))

	print('%-20s: |   %.4f' %('F1 Score out_1', f1(res[28],res[29],res[30])))
	print('%-20s: |   %.4f' %('IOU out_1', mean_iou(res[28],res[29],res[30])))

	plt.subplot(1,3,1)
	plt.imshow(test_x[0])

	plt.subplot(1,3,2)
	tmp = test_y[0]*255
	j = np.dstack((tmp,tmp,tmp)).astype(np.uint8)
	plt.imshow(j)

	plt.subplot(1,3,3)
	arr_pred = model.predict(test_x)
	if isinstance(arr_pred, list):
		arr_pred = arr_pred[0]
	tmp = (arr_pred[0]*255)
	img_pred = np.dstack((tmp,tmp,tmp)).astype(np.uint8)
	print(arr_pred.shape)
	plt.imshow(img_pred)

	# plt.show()