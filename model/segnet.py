# Segnet
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import layers

# ===================================================================================

def convolutional_block(X, f, k, name,p = 'same', s = 1):
	a = Conv2D(filters = f, kernel_size = k, padding = p, strides = s, name = name)(X)
	a = BatchNormalization()(a)
	a = Activation('relu')(a)
	return a
def deconvolutional_block(X, f, k, name,p = 'same', s = 1):
	a = Conv2DTranspose(f, k, padding = p, strides = s, name = name)(X)
	a = BatchNormalization()(a)
	a = Activation('relu')(a)
	return a
def segnet(image_size):

	X_input = Input(image_size)

	l1 = convolutional_block(X_input, 64, 3, name = 'conv1', p = 'same', s = 1)
	l2 = convolutional_block(l1, 64, 3, name = 'conv2', p = 'same', s = 1)
	l3 = MaxPooling2D()(l2)

	l4 = convolutional_block(l3, 128, 3, name = 'conv3', p = 'same', s = 1)
	l5 = convolutional_block(l4, 128, 3, name = 'conv4', p = 'same', s = 1)
	l6 = MaxPooling2D()(l5)

	l7 = convolutional_block(l6, 256, 3, name = 'conv5', p = 'same', s = 1)
	l8 = convolutional_block(l7, 256, 3, name = 'conv6', p = 'same', s = 1)
	l9 = convolutional_block(l8, 256, 3, name = 'conv7', p = 'same', s = 1)
	l10 = MaxPooling2D()(l9)

	l11 = convolutional_block(l10, 512, 3, name = 'conv8', p = 'same', s = 1)
	l12 = convolutional_block(l11, 512, 3, name = 'conv9', p = 'same', s = 1)
	l13 = convolutional_block(l12, 512, 3, name = 'conv10', p = 'same', s = 1)
	l14 = MaxPooling2D()(l13)

	l15 = convolutional_block(l14, 512, 3, name = 'conv11', p = 'same', s = 1)
	l16 = convolutional_block(l15, 512, 3, name = 'conv12', p = 'same', s = 1)
	l17 = convolutional_block(l16, 512, 3, name = 'conv13', p = 'same', s = 1)
	l18 = MaxPooling2D()(l17)
 
	# l18 = Dense(1024, activation = 'relu', name='fc1')(l18)
	# l18 = Dense(1024, activation = 'relu', name='fc2')(l18)

	# ===============================================================================
	l19 = UpSampling2D()(l18)
	l20 = deconvolutional_block(l19, 512, 3, name = 'dconv1', p = 'same', s = 1)
	l21 = deconvolutional_block(l20, 512, 3, name = 'dconv2', p = 'same', s = 1)
	l22 = deconvolutional_block(l21, 512, 3, name = 'dconv3', p = 'same', s = 1)

	l22 = UpSampling2D()(l21)
	l23 = deconvolutional_block(l22, 512, 3, name = 'dconv4', p = 'same', s = 1)
	l24 = deconvolutional_block(l23, 512, 3, name = 'dconv5', p = 'same', s = 1)
	l25 = deconvolutional_block(l24, 512, 3, name = 'dconv6', p = 'same', s = 1)

	l26 = UpSampling2D()(l25)
	l27 = deconvolutional_block(l26, 256, 3, name = 'dconv7', p = 'same', s = 1)
	l28 = deconvolutional_block(l27, 256, 3, name = 'dconv8', p = 'same', s = 1)
	l29 = deconvolutional_block(l28, 256, 3, name = 'dconv9', p = 'same', s = 1)

	l30 = UpSampling2D()(l29)
	l31 = deconvolutional_block(l30, 128, 3, name = 'dconv10', p = 'same', s = 1)
	l32 = deconvolutional_block(l31, 128, 3, name = 'dconv11', p = 'same', s = 1)

	l33 = UpSampling2D()(l32)
	l34 = deconvolutional_block(l33, 64, 3, name = 'dconv12', p = 'same', s = 1)
	l35 = Conv2DTranspose(1, 3, padding = 'same', strides = 1, name = 'dconv13')(l34)
	l36 = BatchNormalization()(l35)
	X_output = Activation('sigmoid')(l36)

	model = Model(X_input, X_output, name = 'SegNet')
	# model.summary()

	return model