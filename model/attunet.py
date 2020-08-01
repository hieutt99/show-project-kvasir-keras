# Attention UNet 

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

def attention_gate(X, X_g, filters):
    x = Conv2D(filters = filters, kernel_size = 1, strides = 1)(X)
    g = Conv2D(filters = filtres, kernel_size = 1, strides = 1)(X_g)

    f = Activation('relu')(add([x,g]))
    psi = Conv2D(1, kernel_size = 1, strides = 1)(f)
    mask = Activation('sigmoid')(psi)

    output = multiply([x,mask])
    return output

def up_concat_block(X, X_cat, filters, size = (2,2)):
	
	up = UpSampling2D(size = size)(X)
	up = Conv2D(filters = filters, kernel_size = size, padding = 'same')(up)
	
	cat = concatenate([X_cat, up])
	
	return cat

def Down(X, filters, kernel_size, padding = 'same', strides = 1):
	down = double_conv_block(X, filters, kernel_size, padding, strides)
	down = MaxPooling2D((2,2))(down)
	return down

def Up(X, X_cat, filters, kernel_size, padding = 'same', strides = 1):
	up = up_concat_block(X, X_cat, filters, size = (2,2))
	up = double_conv_block(up, filters, kernel_size, padding = padding, strides = strides)
	return up


def UNet(image_size = (256,256,3), name = "UNet model"):
	X_input = Input(image_size)

	# encoder
	down1 = Down(X_input, 64, 3)
	
	down2 = Down(down1, 128, 3)
	
	down3 = Down(down2, 256, 3)

	down4 = Down(down3, 512, 3)

	down5 = Down(down4, 1024, 3)

	# decoder
	up1 = Up(down5, down4, 512, 3)

	up2 = Up(up1, down3, 256, 3)

	up3 = Up(up2, down2, 128, 3)

	up4 = Up(up3, down1, 64, 3)

	# filters = # output classes 
	X_output = Conv2D(filters = 1, kernel_size = 1, padding = 'same')(up4)



	model = Model(X_input, X_output, name = name)
	model.summary()

	return model


