# UNet 

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import layers

def conv_block(X, filters, kernel_size, padding = 'same', strides = 1):
	output = Conv2D(filters = filters, kernel_size = kernel_size, padding = padding, strides = strides)(X)
	output = BatchNormalization()(output)
	output = Activation('relu')(output)
	return output

def double_conv_block(X, filters, kernel_size, padding = 'same', strides = 1):
	output = conv_block(X, filters, kernel_size, padding, strides)
	output = conv_block(output, filters, kernel_size, padding, strides)
	return output

def up_concat_block(X, X_cat, filters, size = (2,2)):
	
	up = UpSampling2D(size = size)(X)
	up = Conv2D(filters = filters, kernel_size = size, padding = 'same')(up)
	
	cat = concatenate([X_cat, up])
	
	return cat

def Down(X, filters, kernel_size, padding = 'same', strides = 1):
	down = MaxPooling2D((2,2))(X)
	down = double_conv_block(down, filters, kernel_size, padding, strides)
	return down

def Up(X, X_cat, filters, kernel_size, padding = 'same', strides = 1):
	up = up_concat_block(X, X_cat, filters, size = (2,2))
	up = double_conv_block(up, filters, kernel_size, padding = padding, strides = strides)
	return up


def UNet(image_size = (256,256,3), name = "UNet model"):
	X_input = Input(image_size)

	# encoder
	down1 = double_conv_block(X_input, 64, 3)
	
	down2 = Down(down1, 128, 3)
	
	down3 = Down(down2, 256, 3)

	down4 = Down(down3, 512, 3)

	bottom = Down(down4, 1024, 3)

	# decoder
	dec = Up(bottom, down4, 512, 3)

	dec = Up(dec, down3, 256, 3)

	dec = Up(dec, down2, 128, 3)

	dec = Up(dec, down1, 64, 3)

	# filters = # output classes 
	dec = Conv2D(filters = 2, kernel_size = 3, activation = 'relu', padding = 'same')(dec)
	X_output = Conv2D(filters = 1, kernel_size = 1, activation = 'sigmoid')(dec)


	model = Model(X_input, X_output, name = name)
	model.summary()

	return model


