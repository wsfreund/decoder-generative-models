from .meter_base import ScalarEff, GenerativeEffMeter
from ..misc import *

import tensorflow as tf

class ACF(ScalarEff,GenerativeEffMeter):

  def __init__(self, name = "ACF", start_lag = 1, stop_lag = 2):
    super().__init__(name)
    self.start_lag = start_lag
    self.stop_lag = stop_lag

  def initialize(self, x_data, x_gen = None, xmask_g1 = None, xmask_g2 = None):
    if self.initialized:
      return
    if xmask_g1 is not None or xmask_g2 is not None:
      raise NotImplementedError("ePDF is not currently implemented for masked data")
    self.reset()
    self.acf_data = acf_tf(x_data, start_lag = self.start_lag, stop_lag = self.stop_lag)
    self.initialized = True

  def accumulate(self, xgen, xmask = None ):
    if xmask is not None:
      raise NotImplementedError("ACF is not currently implemented for masked data")
    if self.i > 0:
      raise NotImplementedError("ACF is not able to work with multiple minibatches")
    self.start
    self.acf_gen        = acf_tf(xgen, start_lag = self.start_lag, stop_lag = self.stop_lag)
    self.acf_per_feature = self._compute_stats( self.acf_gen )
    self.total_acf       = tf.math.reduce_mean( self._compute_stats( self.acf_gen ) )
    self.i += 1
    self.stop
    return self.total_acf

  @tf.function
  def _compute_stats(self, acf_gen):
    # NOTE Norm-2 is employed in the original code although norm-1 is mentioned
    # on the paper
    return tf.linalg.norm(tf.math.subtract( acf_gen, self.acf_data), ord = 2, axis = 0 )

  def retrieve(self):
    self.print
    return self.total_acf

  def reset(self):
    super().reset()
    self.total_acf = 0.
    self.acf_gen = []
    self.i = 0

@tf.function
def acf_tf(x, start_lag = 1, stop_lag = 2, dim=(0,1)):
  acf_list = list()
  x = x - tf.math.reduce_mean(x, axis = (0,1) )
  var = tf.math.reduce_variance(x, axis = (0,1) )
  for i in range(start_lag, stop_lag):
    y = tf.math.multiply( x[:, i:], x[:, :-i]) if i > 0 else tf.math.square(x)
    acf_i = tf.math.reduce_mean(y, axis=dim ) / var
    acf_list.append(acf_i)
  if dim==(0,1):
    return tf.stack(acf_list)
  else:
    return tf.concat(acf_list, axis = 1)

