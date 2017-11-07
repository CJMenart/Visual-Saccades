#(hopefully) implements an LSTM for visual saccades using Q-learning as a basis for
#the attention, a (hopefully) theoretically and practically sound technique

#TODO:
#-Move up from fixed number of iterations to trainable stop condition
#-write the code
#-expandable batch size? Believe there are several obstacles to that...

import tensorflow as tf
import eyeball
PATCH_SIZE = 32
NSTEP = 10
STATE_SIZE = 100
#import some sentence encoder

def QLSTM(img,nSteps,vggWeightFile):

	#external inputs
	sentence_encoding = tf.placeholder(tf.float32,[1,10])
	patches = []
	VQAInputs = []
	for step in range(NSTEP):
		patches.append(tf.placeholder(tf.float32,[PATCH_SIZE,PATCH_SIZE]))
			patch_encoding = eyeball.eyeball(img,patch,vggWeightFile)
			patch_encoding = tf.squeeze(tf.reduce_mean(patch_encoding,axis=[1,2]),[1,2])
			VQAInputs = tf.concat((sentence_encoding,patch_encoding),1)

	#lstm variables
	hSize = STATE_SIZE + VQAInputs[0].shape.as_list()[-1]
	Wforget = tf.get_variable(tf.float32,[hSize,STATE_SIZE],name='Wforget')
	Bforget = tf.get_variable(tf.float32,[STATE_SIZE],name='Bforget')
	Wremember = tf.get_variable(tf.float32,[hSize,STATE_SIZE],name='Wremember')
	Bremember = tf.get_variable(tf.float32,[STATE_SIZE],name='Bremember')
	Wprocess = tf.get_variable(tf.float32,[hSize,STATE_SIZE],name='Wprocess')
	Bprocess = tf.get_variable(tf.float32,[STATE_SIZE],name='Bprocess')
	Woutput = tf.get_variable(tf.float32,[hSize,STATE_SIZE],name='Woutput')
	Boutput = tf.get_variable(tf.float32,[STATE_SIZE],name='Boutput')
	
	#initial hidden state
	C = tf.zeros([1,STATE_SIZE])
	h = tf.zeros([1,STATE_SIZE])
	for step in range(NSTEP):
		hidden = tf.concat((VQAInputs[step],h),1)
		forget = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(hidden,Wforget),Bforget))
		C = C*forget
		remember = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(hidden,Wremember),Bremember))
		toRemember = tf.nn.tanh(tf.nn.bias_add(tf.matmul(hidden,Wprocess),Bprocess))
		C = C + toRemember
		outputGate = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(hidden,Woutput),Boutput))
		h = outputGate*tf.nn.tanh(C)
	
	
	
	#unroll the cells
