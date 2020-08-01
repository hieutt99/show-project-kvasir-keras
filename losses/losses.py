from evaluation.metrics import * 

def TverskyLoss(y_true, y_pred):
	return 1 - tversky(y_true,y_pred)

def FocalTverskyLoss(y_true,y_pred):
	pt_1 = tversky(y_true, y_pred)
	gamma = 0.75
	return K.pow((1-pt_1), gamma)

def DiceLoss(y_true, y_pred):
	return 1-dice_coefficient(y_true, y_pred)

def BoundaryLoss(y_true, y_pred):
	pass