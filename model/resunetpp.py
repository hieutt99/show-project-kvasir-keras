from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import layers


# ===================================================================================
def convolutional_block(X, filters, kernel_size, padding = 'same', strides = 1):
	a = BatchNormalization()(X)
	a = Activation('relu')(a)
	a = Conv2D(filters, kernel_size, padding = padding, strides = strides)(a)
	return a

def squeeze_excite_block(inputs, ratio=8):
	init = inputs
	channel_axis = -1
	filters = init.shape[channel_axis]
	se_shape = (1, 1, filters)

	se = GlobalAveragePooling2D()(init)
	se = Reshape(se_shape)(se)
	se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
	se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

	x = Multiply()([init, se])
	return x

def aspp_block(x, num_filters, rate_scale=1):
	x1 = Conv2D(num_filters, (3, 3), dilation_rate=(6 * rate_scale, 6 * rate_scale), padding="SAME")(x)
	x1 = BatchNormalization()(x1)

	x2 = Conv2D(num_filters, (3, 3), dilation_rate=(12 * rate_scale, 12 * rate_scale), padding="SAME")(x)
	x2 = BatchNormalization()(x2)

	x3 = Conv2D(num_filters, (3, 3), dilation_rate=(18 * rate_scale, 18 * rate_scale), padding="SAME")(x)
	x3 = BatchNormalization()(x3)

	x4 = Conv2D(num_filters, (3, 3), padding="SAME")(x)
	x4 = BatchNormalization()(x4)

	y = Add()([x1, x2, x3, x4])
	y = Conv2D(num_filters, (1, 1), padding="SAME")(y)
	return y

def attention_block(g, x):
	"""
		g: Output of Parallel Encoder block
		x: Output of Previous Decoder block
	"""
	filters = x.shape[-1]

	g_conv = convolutional_block(g, filters, kernel_size = 3, padding = "same", strides = 1)
	# g_conv = BatchNormalization()(g)
	# g_conv = Activation("relu")(g_conv)
	# g_conv = Conv2D(filters, (3, 3), padding="SAME")(g_conv)

	g_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(g_conv)

	x_conv = convolutional_block(x, filters, kernel_size = 3, padding = "same", strides = 1)
	# x_conv = BatchNormalization()(x)
	# x_conv = Activation("relu")(x_conv)
	# x_conv = Conv2D(filters, (3, 3), padding="SAME")(x_conv)

	gc_sum = Add()([g_pool, x_conv])

	gc_conv = convolutional_block(gc_sum, filters, kernel_size = 3, padding = "same", strides = 1)
	# gc_conv = BatchNormalization()(gc_sum)
	# gc_conv = Activation("relu")(gc_conv)
	# gc_conv = Conv2D(filters, (3, 3), padding="SAME")(gc_conv)

	gc_mul = Multiply()([gc_conv, x])
	return gc_mul
# stem block added squeeze and excite block 
def stem(X, filters, kernel_size, padding = 'same', strides = 1):
	a = Conv2D(filters, kernel_size, padding  = padding, strides = strides)(X)
	a = convolutional_block(a, filters, kernel_size, padding = padding, strides = strides)

	shortcut = Conv2D(filters, kernel_size = (1,1), padding = padding, strides = strides)(X)
	shortcut = BatchNormalization()(shortcut)

	output = Add()([a, shortcut])
	output = squeeze_excite_block(output)
	return output
# residual block added squeeze and excite block 
def residual_block(X, filters, kernel_size, padding = 'same',dilation_rates = 1, strides = 1):
	a = convolutional_block(X, filters, kernel_size, padding = padding, strides = strides)
	a = convolutional_block(a, filters, kernel_size, padding = padding, strides = 1)
	# Addition
	shortcut = Conv2D(filters, kernel_size = 1, padding = padding, strides = strides)(X)
	shortcut = BatchNormalization()(shortcut)

	output = Add()([shortcut, a])
	output = squeeze_excite_block(output)
	return output

def upsample_concate_block(X, X_shortcut):
	a = UpSampling2D((2,2))(X)
	a = Concatenate()([a,X_shortcut])
	return a



# ==========================================================================================================
def first(X, filters, kernel_size, padding = 'same', strides = 1):
	a = Conv2D(filters, kernel_size, padding  = padding, strides = strides)(X)
	a = convolutional_block(a, filters, kernel_size, padding = padding, strides = strides)

	shortcut = Conv2D(filters, kernel_size = kernel_size, padding = padding, strides = strides)(X)
	shortcut = BatchNormalization()(shortcut)

	output = Add()([a, shortcut])
	return output

# ==============================================================================================

def ResUNetPlusPlus(image_size = (256,256,3), name = "ResUNetPlusPlus"):

	X_input = Input(image_size)

	# Encoder
	l1 = stem(X_input, 32, 3, strides = 1)  
	l2 = residual_block(l1, 64, 3, strides = 2) 
	l3 = residual_block(l2, 128, 3, strides = 2)
	l4 = residual_block(l3, 256, 3, strides  = 2)

	# Bridge

	l5 = aspp_block(l4, num_filters=512)

	# Decoder
	l6 = attention_block(l3, l5)
	l7 = upsample_concate_block(l6, l3) 
	l8 = residual_block(l7, 256, 3, strides = 1) 
	
	l9 = attention_block(l2, l8)
	l10 = upsample_concate_block(l9, l2) 
	l11 = residual_block(l10, 128, 3, strides = 1) 

	l12 = attention_block(l1, l11)
	l13 = upsample_concate_block(l12, l1) 
	l14 = residual_block(l13, 64, 3, strides = 1)

	# output
	l15 = aspp_block(l14, num_filters = 32)
	X_output = Conv2D(1, 1, padding = 'same', activation = 'sigmoid')(l15)

	model = Model(X_input, X_output, name = name)
	# model.summary()

	return model


def ResUNetAlter(image_size = (256,256,3), name = "ResUNetAlter"):
    '''
    Alternative version of ResUNet++ try removing ASPP block for speeding up
    '''
	X_input = Input(image_size)

	# Encoder
	l1 = stem(X_input, 32, 3, strides = 1)  
	l2 = residual_block(l1, 64, 3, strides = 2) 
	l3 = residual_block(l2, 128, 3, strides = 2)
	l4 = residual_block(l3, 256, 3, strides  = 2)

	# Bridge

    l5 = convolutional_block(l4, 512, 3, strides = 2) 
	l6 = convolutional_block(l5, 512, 3, strides = 1)

	# Decoder
	l7 = attention_block(l3, l6)
	l8 = upsample_concate_block(l7, l3) 
	l9 = residual_block(l8, 256, 3, strides = 1) 
	
	l10 = attention_block(l2, l9)
	l11 = upsample_concate_block(l10, l2) 
	l12 = residual_block(l11, 128, 3, strides = 1) 

	l13 = attention_block(l1, l12)
	l14 = upsample_concate_block(l13, l1) 
	l15 = residual_block(l14, 64, 3, strides = 1)

	# output
	l15 = aspp_block(l14, num_filters = 32)
	X_output = Conv2D(1, 1, padding = 'same', activation = 'sigmoid')(l15)

	model = Model(X_input, X_output, name = name)
	model.summary()

	return model