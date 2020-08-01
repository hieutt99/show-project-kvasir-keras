from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import layers


# ===================================================================================
def convolutional_block(X, filters, kernel_size, padding = 'same', strides = 1):
	a = BatchNormalization()(X)
	a = Activation('relu')(a)
	a = Conv2D(filters, kernel_size, padding = padding, strides = strides)(a)
	return a

def stem(X, filters, kernel_size, padding = 'same', strides = 1):
	a = Conv2D(filters, kernel_size, padding  = padding, strides = strides)(X)
	a = convolutional_block(a, filters, kernel_size, padding = padding, strides = strides)

	shortcut = Conv2D(filters, kernel_size = (1,1), padding = padding, strides = strides)(X)
	shortcut = BatchNormalization()(shortcut)

	output = Add()([a, shortcut])
	return output

def residual_block(X, filters, kernel_size, padding = 'same',dilation_rates = 1, strides = 1):
	a = convolutional_block(X, filters, kernel_size, padding = padding, strides = strides)
	a = convolutional_block(a, filters, kernel_size, padding = padding, strides = 1)
	# Addition
	shortcut = Conv2D(filters, kernel_size = 1, padding = padding, strides = strides)(X)
	shortcut = BatchNormalization()(shortcut)

	output = Add()([shortcut, a])
	return output

def upsample_concate_block(X, X_shortcut):
	a = UpSampling2D((2,2))(X)
	a = Concatenate()([a,X_shortcut])
	return a

# ==============================================================================================


def first(X, filters, kernel_size, padding = 'same', strides = 1):
	a = Conv2D(filters, kernel_size, padding  = padding, strides = strides)(X)
	a = convolutional_block(a, filters, kernel_size, padding = padding, strides = strides)

	shortcut = Conv2D(filters, kernel_size = kernel_size, padding = padding, strides = strides)(X)
	shortcut = BatchNormalization()(shortcut)

	output = Add()([a, shortcut])
	return output

def ResUNet(image_size = (256,256,3), name = "ResUNet model"):

	X_input = Input(image_size)

	# Encoder
	l1 = first(X_input, 32, 3, strides = 1) 
	l2 = residual_block(l1, 64, 3, strides = 2) 
	l3 = residual_block(l2, 128, 3, strides = 2) 
	l4 = residual_block(l3, 256, 3, strides  = 2) 

	# Bridge

	l5 = convolutional_block(l4, 512, 3, strides = 2) 
	l6 = convolutional_block(l5, 512, 3, strides = 1)

	# Decoder

	l7 = upsample_concate_block(l6, l4) 
	l8 = residual_block(l7, 256, 3, strides = 1)
	
	l9 = upsample_concate_block(l8, l3) 
	l10 = residual_block(l9, 128, 3, strides = 1) 
	
	l11 = upsample_concate_block(l10, l2) 
	l12 = residual_block(l11, 64, 3, strides = 1) 

	l13 = upsample_concate_block(l12, l1)
	l14 = residual_block(l13, 32, 3, strides = 1) 

	X_output = Conv2D(1, 1, padding = 'same', activation = 'sigmoid')(l14)

	model = Model(X_input, X_output, name = name)
	model.summary()

	return model