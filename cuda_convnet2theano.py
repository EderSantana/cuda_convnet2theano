import cPickle
import convnet
import theano
from theano import tensor as T
import numpy as np
import itertools
import climin.util
from my_rmsprop import RmsProp, GradientDescent

from theano.tensor.nnet.conv import conv2d as cpu_conv
#from theano.tensor.signal.downsample import max_pool_2d

from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.pool import MaxPool
from pylearn2.sandbox.cuda_convnet.response_norm import CrossMapNorm
#from theano_linear.unshared_conv.gpu_unshared_conv import GpuFilterActs


def conv2d(X,W,padding=0,pad=0,stride=1,parsum=1):
  pad = np.abs(pad)
  if theano.config.device == 'gpu':
    conv_op = FilterActs(partial_sum=parsum, pad=pad, stride=stride)
    Xcont = gpu_contiguous(X.dimshuffle(1,2,3,0))
    Wcont = gpu_contiguous(W.dimshuffle(1,2,3,0))
    out   = conv_op(Xcont, Wcont).dimshuffle(3,0,1,2)
  else:
    out = cpu_conv(X, W)
  return out

'''
def local2d(X,W):
  local_op = GpuFilterActs(1)
  Xcont = gpu_contiguous(X.dimshuffle(1,2,3,0))
  Wcont = gpu_contiguous(W.dimshuffle(1,2,3,0))
  out   = local_op(Xcont, Wcont).dimshuffle(3,0,1,2)
  return out
'''

def maxpool(X, size=3, stride=2):
  pool_op = MaxPool(ds=size, stride=stride)
  contiguous_input = gpu_contiguous(X.dimshuffle(1,2,3,0))
  out = pool_op(contiguous_input).dimshuffle(3,0,1,2)
  return out

def cmrnorm(X, size=5, alpha=.001, beta=.75):
  cnorm_op = CrossMapNorm(size_f=size, add_scale=alpha, pow_scale=beta, blocked=False)
  contiguous_input = gpu_contiguous(X.dimshuffle(1,2,3,0))
  out = T.as_tensor(cnorm_op(contiguous_input)).sum(axis=0).dimshuffle(3,0,1,2)
  return out

class TheanoCNN():

  def __init__(self, path, output_layer, number_of_classes, batches):
    self.batches = batches
    self.y, self.output, self.W, self.b = self._get_cnn(path, output_layer)
    self.xgrad = []
    for i in range(number_of_classes):
      self.xgrad += [ T.grad(self.y[:,i].sum(), self.X) ]
    self.saliency = T.grad(self.y.argmax(axis=1).sum(), self.X)
    return self

  def _get_cnn(self,path,output_layer):
    f = file(path,'r')
    cnn = cPickle.load(f)
    f.close()
    isfirst = True
    try:
      L = cnn['model_state']['layers'] # Cuda-convnet CNN
    except:
      L = cnn # Decaf-convnet

    was_previous_conv = False
    W = []
    b = []
    for l in L:
      print "Loading layer " + l['name']
      if l['type']=='conv':
        if isfirst:
          X = T.tensor4()
          y = X
          isfirt = False
        ch = l['channels'][0]
        sh = np.sqrt(l['filterPixels'])
        w =  theano.shared(l['weights'][0].reshape((ch,sh,sh,-1)).transpose(3,0,1,2))
        W += [w]
        b += [theano.shared(l['biases'][:,0])]
        y = conv2d(y, w, pad=l['padding'][0], stride=l['stride'], parsum=l['partialSum']) + b[-1].dimshuffle('x',0,'x','x')
        was_previous_conv = True
      elif l['type']=='fc':
        if isfirst:
          X = T.matrix()
          y = X
          isfirt = False
        if was_previous_conv:
          y = y.reshape((self.batches, -1))
        W += [theano.shared(l['weights'][0])]
        b += [theano.shared(l['biases'][0,:])]
        y = T.dot(y, W) + b[-1]
        was_previous_conv = False
      elif l['type'] == 'neuron':
        if l['neuron']['type']=='relu':
          y = T.maximum(0.0,y)
          was_previous_conv = True
      elif l['type'] == 'pool':
        y = maxpool(y, size=l['sizeX'],stride=l['stride'])
        was_previous_conv = True
      elif l['type'] == 'cmrnorm':
        y = cmrnorm(y, size=l['size'], alpha=l['scale'], beta=l['pow'])
        was_previous_conv = True

      if l['name'] == output_layer:
        yout = y

    return y, yout, W, b

  def compile(self):
    self.fxgrad = []
    for i in range(number_of_classes):
      self.fxgrad += [theano.function( self.xgrad[i], self.X, allow_input_downcast=True ) ]
    self.fsaliency = theano.function( self.saliency, X, allow_input_downcast=True )
    self.foutput = theano.function( self.output, X, allow_input_downcast=True)
    return self
