__author__ = "Tran Trung Hieu"
# __modelName__ = "SUNet_model_xxx"
__modelName__ = "SUNet_DSPVS_model_xxx"
# ========================================================================
from model.resunet import *
from model.unet import * 
from model.sunet import *


from losses.losses import *
from evaluation.metrics import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import *
import os
from evaluation.model_eval import *

import argparse
# ========================================================================
# arguments

img_h = 256
img_w = 256

seed = 1337
test_seed = 1338
val_seed = 1339

batch_size = 8

data_gen_args = dict(featurewise_center=False,
		featurewise_std_normalization=False,
		rotation_range = 90, 
		width_shift_range = 0.2,
		height_shift_range = 0.2,
		zoom_range = 0.2,	
		shear_range = 0.2,
		horizontal_flip = True,
		rescale = 1./255)

# train_images_folder = '/content/kvasir/train'
# train_masks_folder = '/content/kvasir/train'

# test_images_folder = '/content/kvasir/test'
# test_masks_folder = '/content/kvasir/test'

# val_images_folder = '/content/kvasir/val'
# val_masks_folder = '/content/kvasir/val'

train_images_folder = './train/'
train_masks_folder = './train/'

test_images_folder = './test/'
test_masks_folder = './test/'

val_images_folder = './val/'
val_masks_folder = './val/'
# ================================================================================
# model_save_path = './resunetpp_colab_1.h5'
model_save_path = './sunet_colab_dspvs_2.h5'
# ================================================================================
parser = argparse.ArgumentParser()

# ============================================================================================
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

test_image_datagen = ImageDataGenerator(rescale = 1./255)
test_mask_datagen = ImageDataGenerator(rescale = 1./255)

val_image_datagen = ImageDataGenerator(rescale = 1./255)
val_mask_datagen = ImageDataGenerator(rescale = 1./255)
# ===================================================================
val_image_generator = val_image_datagen.flow_from_directory(directory = val_images_folder, \
	target_size = (img_h,img_w), batch_size = batch_size, shuffle = True, color_mode = 'rgb', classes = ['images'], class_mode = None, seed = val_seed)
val_mask_generator = val_mask_datagen.flow_from_directory(directory = val_masks_folder, \
	target_size = (img_h,img_w), batch_size = batch_size, shuffle = True, color_mode = 'grayscale', classes = ['masks'], class_mode = None, seed = val_seed)

val_generator = zip(val_image_generator, val_mask_generator)
# ===================================================================
test_image_generator = test_image_datagen.flow_from_directory(directory = test_images_folder, \
	target_size = (img_h,img_w), batch_size = batch_size, shuffle = True, color_mode = 'rgb', classes = ['images'], class_mode = None, seed = test_seed)
test_mask_generator = test_mask_datagen.flow_from_directory(directory = test_masks_folder, \
	target_size = (img_h,img_w), batch_size = batch_size, shuffle = True, color_mode = 'grayscale', classes = ['masks'], class_mode = None, seed = test_seed)

test_generator = zip(test_image_generator, test_mask_generator)
# ===================================================================
image_generator = image_datagen.flow_from_directory(directory = train_images_folder, \
	target_size = (img_h,img_w), batch_size = batch_size, shuffle = True, color_mode = 'rgb', classes = ['images'], class_mode = None, seed = seed)
mask_generator = mask_datagen.flow_from_directory(directory = train_masks_folder, \
	target_size = (img_h,img_w), batch_size = batch_size, shuffle = True, color_mode = 'grayscale', classes = ['masks'], class_mode = None, seed = seed)

train_generator = zip(image_generator, mask_generator)
# ================================================================================


def checkDevice():
	from tensorflow.python.client import device_lib
	device_lib.list_local_devices()

def train(model, model_save_path = model_save_path, \
	epochs = 10, verbose = 1, train_generator = train_generator, steps_per_epoch = 2000, \
	val_data = val_generator, val_steps = 12):
	print("============================================================")
	print("--------------Training progress-----------------------------")
	print("------------------------------------------------------------")
	history = model.fit_generator(generator = train_generator, steps_per_epoch = steps_per_epoch,\
		epochs = epochs, verbose = verbose, validation_data = val_generator, validation_steps = val_steps)
	model.save(model_save_path)
	return history

def initOptimizer(args = 'nadam', lr = 0.0001):
	if args == 'nadam':
		nadam = Nadam(lr = lr)
		return  nadam
	if args == 'adam':
		adam = Adam(lr = lr)
		return adam

def main():
	checkDevice()
	print("============================================================")
	# resunet = ResUNet((img_h,img_w, 3),name = __modelName__)
	# model = UNet((img_h,img_w, 3),name = __modelName__)
	 model = SUNet((img_h, img_w, 3), name = __modelName__)
	
	# model.compile(optimizer = initOptimizer(), loss = DiceLoss,\
	# 	metrics = [tversky, dice_coefficient, iou, precision, recall, accuracy, true_positive, false_negative, false_positive])
	
	# deep supervision
	model.compile(optimizer = initOptimizer(), loss = [DiceLoss, DiceLoss, DiceLoss],\
		metrics = [tversky, dice_coefficient, iou, precision, recall, accuracy, true_positive, false_negative, false_positive])
	
	if os.path.exists(model_save_path):
		model.load_weights(model_save_path)
	history = train(model)
	print("Exiting...")
	return model
def eval(model, train_generator, val_generator, test_generator):
	model_evaluate(model, train_generator, val_generator, test_generator, \
    batch_size = 8, train_size = 700, val_size = 100, test_size = 200)
	testSetReveal(model, test_generator)

if __name__ == "__main__":
	# print(__author__)
	model = main()
	eval(model, train_generator, val_generator, test_generator)


