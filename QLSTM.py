#(hopefully) implements an LSTM for visual saccades using Q-learning as a basis for
#the attention, a (hopefully) theoretically and practically sound technique

#TODO:
#-Move up from fixed number of iterations to trainable stop condition
#-write the code
#-expandable batch size? Believe there are several obstacles to that...

import tensorflow as tf
import eyeball
import numpy as np
from random import shuffle

PATCH_SIZE = 32
NSTEP = 10
STATE_SIZE = 100
COORD_RES = 25 #COORD_RES + PATCH_SIZE <= smallest image dimension in dataset. No safety checks as of yet!
COORD_LAYER_SIZES = [250,500] #size of hidden layers used to select coordinates
ANS_LAYER_SIZES = [250,500,1000] #size of layers used to give answer
NEPS = 500
NUM_CLASSES = 1000
#import some sentence encoder

#constructs a network, inception-style again, such that we can implement RL however we wish b/c it manually
#unrolls all the stuff
def QLSTM(img,sentence_encoding,nSteps,vggWeightFile):

	#external inputs
	#sentence_encoding = tf.placeholder(tf.float32,[1,10]) #this should change...
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
	#for x,y
	coordWeights = []
	coordBiases = []
	for lay in range(len(COORD_LAYER_SIZES)):
		sz = COORD_LAYER_SIZES[lay]
		if lay==0:
			cIn = STATE_SIZE
		else:
			cIn = COORD_LAYER_SIZES[lay-1]
		coordWeights.append(tf.get_variable('coordW%d' % lay),shape=[cIn,sz],initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2/cIn)))
		coordBiases.append(tf.get_variable('coordB%d' % lay,shape[sz],initializer=tf.constant_initializer(0.01)))
	coordWeights.append(tf.get_variable('coordW%d' % lay+1),shape=[COORD_LAYER_SIZES[-1],COORD_RES*COORD_RES],initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2/COORD_LAYER_SIZES[-1])))
	coordBiases.append(tf.get_variable('coordB%d' % lay+1,shape[COORD_RES*COORD_RES],initializer=tf.constant_initializer(0.01)))
	
	#initial hidden state
	C = tf.zeros([1,STATE_SIZE])
	h = tf.zeros([1,STATE_SIZE])
	#unroll the cells
	coordSelectors = []	
	hiddenStates = []
	Qtrainers = []
	for step in range(NSTEP):
		hidden = tf.concat((VQAInputs[step],h),1)
		forget = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(hidden,Wforget),Bforget))
		C = C*forget
		remember = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(hidden,Wremember),Bremember))
		toRemember = tf.nn.tanh(tf.nn.bias_add(tf.matmul(hidden,Wprocess),Bprocess))
		C = C + toRemember
		outputGate = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(hidden,Woutput),Boutput))
		h = outputGate*tf.nn.tanh(C)
		#x,y coordinate defined as a function of h
		hiddenStates.append(h)
		cFeat = tf.stop_gradient(h)
		for lay in range(len(COORD_LAYER_SIZES)):
			cFeat = tf.nn.leaky_relu(tf.nn.bias_add(tf.matmul(cFeat,coordWeights[lay]),coordBiases[lay]))
		coordSelectors.append(tf.nn.bias_add(tf.matmul(cFeat,coordWeights[-1]),coordBiases[-1]))
		Qtarg = tf.placeholder(tf.float32,[None,COORD_RES*COORD_RES])
		Qtrainers.append((Qtarg,tf.losses.softmax_cross_entropy(coordSelectors[-1],Qtarg)))
		
	#if you change to variable episode length, move this into loop
	#network for giving answer
	aFeat = h
	for lay in range(len(ANS_LAYER_SIZES)):
		sz = ANS_LAYER_SIZES[lay]
		if lay==0:
			cIn = STATE_SIZE
		else:
			cIn = ANS_LAYER_SIZES[lay-1]
		aWeights = tf.get_variable('ansW%d' % lay,shape=[cIn,sz],initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2/cIn))
		aBiases = tf.get_variable('ansB%d' % lay,shape[sz],initializer=tf.constant_initializer(0.01))
		feat = tf.nn.bias_add(tf.matmul(aFeat,aWeights),aBiases)
		aFeat = tf.nn.leaky_relu(feat)
	aWeights = tf.get_variable('ansW%d' % lay+1,shape=[cIn,sz],initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2/cIn))
	aBiases = tf.get_variable('ansB%d' % lay+1,shape[sz],initializer=tf.constant_initializer(0.01))
	answer = tf.nn.bias_add(tf.matmul(aFeat,aWeights),aBiases)
	
	target = tf.placeholder(tf.float32,[None,NUM_CLASSES)
	loss = tf.losses.softmax_cross_entropy(answer,target)
	
	return (patches,coordSelectors,answer,target,answerLoss,Qtrainers)
	
#incrementally roll image through the network, can be used to train or to generate 'episodes' for Q-learning?
#TODO: Probaly need to add sentence here to add to feed_dict
def forward_pass(image,img,patches,coordSelectors,hiddenStates,answer,sess,train):
	nSteps = len(patches)
	sz = image.shape
	avlSz = image.shape - PATCH_SIZE
	h = sess.partial_run_setup(coordSelectors+hiddenStates+[answer],[img]+patches)
	#default for now, first patch is middle of network
	x = int(COORD_RES/2)
	y = x
	coordVals = []
	hiddenVals = []
	for step in range(nSteps);
		#select patch
		xAc = avlSz[1]*y/(COORD_RES-1)
		yAc = avlSz[0]*x/(COORD_RES-1)
		inPatch = img[yAc:yAc+PATCH_SIZE,xAc:xAc+PATCH_SIZE,:]
		#run
		coords,h = sess.partial_run(h,[coordSelectors[step],hiddenStates[ste]],feed_dict={patches[step]:inPatch}) #TODO: Add sentence rep?
		coordVals.append(coords)
		hiddenVals.append(h)
		#turn coords into x,y
		c = np.argmax(coords)
		y = np.floor(c/COORD_RES)
		x = c - y
	ans = sess.partial_run(h,answer)
	
	#backward pass
	if train:
		sess.partial_run(h,train)
	
	return (ans,coordVals,hiddenVals)
		
def collectQEps(image,img,patches,coordSelectors,answer,sess,hiddenStates):
	episodes = []
	
	for run in range(NEPS):
		ans,coordVals,hiddenVals = forward_pass(image,img,patches,coordSelectors,answer,sess,None)
		for ep in range(len(coordVals)):
			#NOTE: Will base case have trouble b/c decision has no effect?
			if ep == len(coordVals)-1:
				loss = sess.run(answerLoss,feed_dict={hiddenStates[ep]:hiddenVals[ep]}) #will this broadcast automatically? Likely source of error
				episode = (hiddenVals[ep],loss)
			else:
				loss = sess.run(coordSelectors[ep+1],feed_dict={hiddenStates[ep+1]:hiddenVals[ep+1]})
				episode = (hiddenVals[ep],loss)
		episodes.append(episode)
	
	shuffle(episodes)
	return episodes

#collect eps, then train for a bit
def QTrain(image,img,patches,coordSelectors,answer,sess,hiddenStates,Qtrainers):
	episodes = collectQEps(image,img,patches,coordSelectors,answer,sess,hiddenStates)
	for ep in len(episodes):
		episode = episodes[ep]
		sess.run(Qtrainers[0],feed_dict={hiddenStates[0]:episodes[ep][0],goodCoords:episodes[ep][1]})