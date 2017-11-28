'''
A tight Convolutional Neural Net meant to operate on 32x32 or 64x64 image patches for
the Visual Saccades Project.
Defined in Inception style. Output of eyeball() is the output TF node.
Im is a 1xNonexNonex3 Tensor. patch is an image-patch tensor of FIXED DIMENSIONALITY.
It may seem slightly redundant to be also passing patch when conceptually it is just a cropping of
im, but there are several ways to determine patch and I don't want this file responsible for them.
vggWeightFile string path to vgg16_weights.npz
~Chris Menart, 10-21-17
'''

#NB: Not yet tested in TF 1.4, only 1.2
import tensorflow as tf
import vgg16Partial
import numpy as np

TRAIN_PERIPHERAL = False
NUM_FOVEAL_CONV = 6
FOVEAL_CHANN = 256
RESIDUAL_FOVEA = False
L2_REG = 1e-3

def peripheral(im,vggWeightFile):
	vggLayers = vgg16Partial.vgg16(im,vggWeightFile,TRAIN_PERIPHERAL,fov[0])
	peripheral = vggLayers.out
	return peripheral

def foveal(patch,bNormFunc):
	fov = patch.shape.as_list()[1:3] #x,y dimensions of patch. We assume it is square
	assert fov[0] == fov[1]
	
	with tf.name_scope('cnn') as scope:
		#'peripheral vision features'
		#frozen partial VGG-Net with a spatial pooling down to feature size
		#currently 128 features
		#note that if you turn on training, there is currently no regularization
		
		#'foveal' vision features, aka high-resolution important part
		#note that x/y dimension is never changed
		# this part MUST be trainable, so take all precautions to make it easy to train!
		# architecture may not be optimal yet--a way to use pre-trained networks would be great
		foveal = patch
		for c in range(NUM_FOVEAL_CONV):
			with tf.variable_scope('fov_conv_%d' % c) as scope:
				in_channel = foveal.shape.as_list()[3]
				kernel = tf.get_variable('weights',shape=[3, 3, in_channel, FOVEAL_CHANN],
					initializer=tf.truncated_normal_initializer(mean=0.0,stddev=np.sqrt(2/(3*3*in_channel)),dtype=tf.float32),
					regularizer=tf.contrib.layers.l2_regularizer(L2_REG))
				biases = tf.get_variable('biases',shape=[FOVEAL_CHANN],initializer=tf.constant_initializer(0.01, dtype=tf.float32),
									 trainable=True)
				
				conv = tf.nn.conv2d(foveal, kernel, [1, 1, 1, 1], padding='SAME')				
				linearSum = tf.nn.bias_add(conv, biases)
				activation = tf.nn.leaky_relu(linearSum)
				
				#NOTE-post-activation norm as suggested by reddit
				out = bNormFunc(activation)
				
				#maybe you want to try a residual net? Theoretical results suggest easier training
				if RESIDUAL_FOVEA and in_channel == FOVEAL_CHANN:
					out = foveal + out
				foveal = out
				
	return foveal