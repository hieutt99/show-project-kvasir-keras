# SUNet 

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import layers



# ================================================================================
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


def UNet(image_size = (256,256,3)):
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
# ================================================================================
def residual_block(X, filters, kernel_size, padding = 'same', strides = 1):
	a = double_conv_block(X, filters)
	# Addition
	shortcut = Conv2D(filters, kernel_size = 1, padding = padding, strides = strides)(X)
	shortcut = BatchNormalization()(shortcut)

	output = Add()([shortcut, a])
	return output

def unet_module(X, in_f, out_f, n_levels):
	l1 = conv_block(X, filters = in_f, kernel_size=1, strides = 1)
	
	down1 = double_conv_block(l1, 64, 3)

	down_list = [down1]
	
	k = 64
	for i in range(n_levels):
		k = k*2
		temp = Down(down_list[-1], k, 3)
		down_list.append(temp)
	# bottom
	k = k*2
	bottom = Down(down_list[-1], k, 3)
	# decoder
	k = k//2
	temp = down_list.pop(-1)
	dec = Up(bottom, temp, k, 3)
	for i in range(n_levels):
		k = k//2
		temp = down_list.pop(-1)
		dec = Up(dec, temp, k, 3)

	output = Conv2D(filters = out_f, kernel_size = 1, strides = 1)(dec) 
	return output


## mah original prototype

# def SUNet(image_size = (256,256,3), name = "SUNet model"):
# 	X_input = Input(image_size)

# 	l = unet_module(X_input, 16, 16,1)
# 	l = unet_module(l, 16, 16, 2)
# 	l = unet_module(l, 16, 16, 1)

# 	l = Conv2D(filters = 2, kernel_size = 3, activation = 'relu', padding = 'same')(l)
# 	X_output = Conv2D(filters = 1, kernel_size = 1, activation = 'sigmoid')(l)

# 	model = Model(X_input, X_output, name = name)
# 	model.summary()

# 	return model


def SUNet(image_size = (256,256,3), name = "SUNet model"):
	# deep supervision

	X_input = Input(image_size)

	l1 = unet_module(X_input, 16, 16,0)
	l2 = unet_module(l1, 16, 16, 1)
	l3 = unet_module(l2, 16, 16, 2)

	l4 = Conv2D(filters = 2, kernel_size = 3, activation = 'relu', padding = 'same')(l3)
	X_output = Conv2D(filters = 1, kernel_size = 1, activation = 'sigmoid')(l4)

	out_1 = Conv2D(filters = 2, kernel_size = 3, activation = 'relu', padding = 'same')(l1)
	out_1 = Conv2D(filters = 1, kernel_size = 1, activation = 'sigmoid')(out_1)

	out_2 = Conv2D(filters = 2, kernel_size = 3, activation = 'relu', padding = 'same')(l2)
	out_2 = Conv2D(filters = 1, kernel_size = 1, activation = 'sigmoid')(out_2)

	model = Model(X_input, [X_output, out_2, out_1], name = name)
	model.summary()

	return model
