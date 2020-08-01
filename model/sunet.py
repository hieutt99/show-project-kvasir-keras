# SUNet 

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import layers

def conv_block(X, filters, kernel_size, padding = 'same', strides = 1):
	output = Conv2D(filters = filters, kernel_size = kernel_size, padding = padding, strides = strides)(X)
	output = BatchNormalization()(output)
	output = Activation('relu')(output)
	return output

def deconv_block(X, filters, kernel_size, padding = 'same', strides = 1):
	output = Conv2DTranspose(filters = filters, kernel_size = kernel_size, padding = padding, strides = strides)(X)
	output = BatchNormalization()(output)
	output = Activation('relu')(output)
	return output

def double_conv_block(X, filters):
	x = conv_block(X, filters = filters, kernel_size=3, strides = 2)
	x = conv_block(x, filters = filters, kernel_size=3, strides = 1)
	return x

def residual_block(X, filters, kernel_size, padding = 'same', strides = 1):
	a = double_conv_block(X, filters)
	# Addition
	shortcut = Conv2D(filters, kernel_size = 1, padding = padding, strides = strides)(X)
	shortcut = BatchNormalization()(shortcut)

	output = Add()([shortcut, a])
	return output

def unet_module(X, in_f, out_f):
	l1 = conv_block(X, filters = in_f, kernel_size=1, strides = 1)
	
	l2 = double_conv_block(l1, in_f)

	l3 = double_conv_block(l2, in_f*2)

	l4 = deconv_block(l3, filters = in_f*2, kernel_size=3, strides = 2)

	l5 = concatenate([l4, l2])

	l6 = deconv_block(l5, filters = in_f, kernel_size=3, strides = 2)

	output = Add()([l6, l1])
	output = Conv2D(filters = out_f, kernel_size = 1, strides = 1)(output) 
	return output
	
def transition_block(X):
	output = AveragePooling2D(pool_size=(2,2), strides = 2)(X)
	return output
def SUNet(image_size = (256,256,3), name = "SUNet model"):
	X_input = Input(image_size)
	in_features = [64, 128]
	out_features = [256, 512, 768, 1024]

	l1 = Conv2D(filters = in_features[0], kernel_size = 7, strides = 2, padding = 'same')(X_input)
	
	l2 = residual_block(l1, filters = in_features[1], kernel_size = 3, strides = 2)
	
	l3 = unet_module(l2, in_features[1], out_features[0])
	l4 = unet_module(l3, in_features[1], out_features[0])

	l5 = transition_block(l4)

	l6 = unet_module(l5, in_features[1], out_features[1])
	l7 = unet_module(l6, in_features[1], out_features[1])
	# l8 = unet_module(l7, in_features[0], out_features[1])
	# l9 = unet_module(l8, in_features[0], out_features[1])
	
	# l10 = transition_block(l9)
	l10 = transition_block(l7)

	l11 = unet_module(l10, in_features[1], out_features[2])
	l12 = unet_module(l11, in_features[1], out_features[2])
	# l13 = unet_module(l12, in_features[0], out_features[2])
	# l14 = unet_module(l13, in_features[0], out_features[2])

	# l15 = transition_block(l14)
	l15 = transition_block(l12)

	l16 = unet_module(l15, in_features[1], out_features[3])
	l17 = unet_module(l16, in_features[1], out_features[3])
	# l18 = unet_module(l17, in_features[0], out_features[3])
	# l19 = unet_module(l18, in_features[0], out_features[3])
	
	# l20 = BatchNormalization()(l19)
	l20 = BatchNormalization()(l17)
	
	output = AveragePooling2D(pool_size=(7,7), strides = 1)(l20)
	print(output.shape)
	X_output = Conv2D(1, 1, strides = 1, activation='sigmoid')(output)

	model = Model(X_input, X_output, name = name)
	model.summary()

	return model
