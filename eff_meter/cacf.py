from .meter_base import GenerativeEffMeter
from ..misc import *

import tensorflow as tf
import numpy as np

class CACFMeter(GenerativeEffMeter):

  def __init__(self, name = "cACF", start_lag = 0, stop_lag = 1):
    super().__init__(name)
    self.start_lag = start_lag
    self.stop_lag = stop_lag

  def initialize(self, x_data, x_gen = None, xmask_g1 = None, xmask_g2 = None):
    if self.initialized:
      return
    if xmask_g1 is not None or xmask_g2 is not None:
      raise NotImplementedError("ePDF is not currently implemented for masked data")
    self.reset()
    self.cacf_data = cacf_tf(x_data, start_lag = self.start_lag, stop_lag = self.stop_lag)
    self.initialized = True

  def accumulate(self, x_gen, xmask = None ):
    if xmask is not None:
      raise NotImplementedError("ACF is not currently implemented for masked data")
    if self.i > 0:
      raise NotImplementedError("ACF is not able to work with multiple minibatches")
    self.start
    # TODO test if tf is faster
    self.cacf_gen = cacf_tf(x_gen, start_lag = self.start_lag, stop_lag = self.stop_lag)
    self.total_cacf = self._compute_stats( self.cacf_gen )
    self.i += 1
    self.stop
    return self.total_cacf

  def _compute_stats(self, cacf_gen):
    # NOTE original code does not divide by n
    return tf.reduce_mean( tf.linalg.norm(tf.math.subtract( cacf_gen, self.cacf_data), ord = 1, axis = (0,1) ) )

  def retrieve(self):
    self.print
    return self.total_cacf

  def reset(self):
    super().reset()
    self.total_cacf = 0.
    self.cacf_gen = []
    self.i = 0

@tf.function
def cacf_tf(x, start_lag = 1, stop_lag = 2, comp_corr = True):
  x = tf.math.subtract(x, tf.math.reduce_mean(x, axis=(0,1), keepdims=True))
  if comp_corr:
    x = tf.math.divide_no_nan( x, tf.math.reduce_std(x, axis=(0,1), keepdims=True) )
  cacf_list = list()
  for i in range(start_lag, stop_lag):
    # TODO Improve computation efficiency to spare symmetric computations
    #x_l = tf.linalg.LinearOperatorLowerTriangular(x[:, i:])
    #x_r = tf.linalg.LinearOperatorLowerTriangular(x[:, :-i])
    y = tf.divide( tf.tensordot( x[:, i:], (x[:, :-i] if i > 0 else x), axes=[[0,1],[0,1]] )
                 , tf.cast( tf.multiply((x.shape[1]-i), x.shape[0]), tf.float32 ) )
    #cacf_i = tf.math.reduce_mean(y, axis=1)
    cacf_list.append(tf.expand_dims(y, axis = 2))
  cacf = tf.concat(cacf_list, axis = 2)
  return cacf # NOTE original code used some like "tf.reshape(cacf,(cacf.shape[0], -1, ?))"
