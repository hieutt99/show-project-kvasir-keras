import tensorflow.keras as keras
import keras.backend as K


# Tversky - focal tversky
def tversky(y_true, y_pred, smooth = 0):
	y_true_pos = K.flatten(y_true)
	y_pred_pos = K.flatten(y_pred)
	true_pos = K.sum(y_true_pos * y_pred_pos)
	false_neg = K.sum(y_true_pos * (1-y_pred_pos))
	false_pos = K.sum((1-y_true_pos)*y_pred_pos)
	alpha = 0.55
	return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# IOU - Intersection over Union
def iou(y_true, y_pred):
	intersection = K.flatten(K.abs(y_true * y_pred))
	union = K.clip(K.flatten(y_true) + K.flatten(y_pred),0,1)
	return (K.sum(intersection) + K.epsilon()) / ( K.sum(union) + K.epsilon())
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Precision
# whether the cost of False Positive is too high
# Prec = TP/(TP+FP)
def precision(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Recall 
# Amount of True Positive
# Rec = TP/(TP+FN)
def recall(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def true_positive(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	return true_positives

def false_positive(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	false_positives = K.sum(y_pred) - true_positives
	return false_positives

def false_negative(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))	
	false_negatives = K.sum(y_true) - true_positives
	return false_negatives
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Accuracy
def accuracy(y_true, y_pred):
	'''Calculates the mean accuracy rate across all predictions for binary
	classification problems.
	'''
	return K.mean(K.equal(y_true, K.round(y_pred)))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
# Dice coefficient - F1 Score
# Dice = (2*TP)/((TP+FP)+(TP+FN))
def dice_coefficient(y_true, y_pred, smooth = 0):
	y_true_flatten = K.flatten(y_true)
	y_pred_flatten = K.flatten(y_pred)
	intersection = K.sum(y_true_flatten * y_pred_flatten)
	return (2 * intersection + smooth) / (K.sum(y_true_flatten)+K.sum(y_pred_flatten)+smooth)

def f1(tp, fn, fp):
	return (2*tp)/(2*tp + fp +fn)

def mean_iou(tp, fn, fp):
	return tp/(tp+fp+fn)