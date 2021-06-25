import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

try:
  from misc import *
except ImportError:
  from .misc import *

class MaskModel(object):
  """
  Mask is [0.,,1.]^{n,...} tensor with same shape as the original data
  indicating whether data is available (i.e. not missing) for that input.
  In other words, a 1. flag indicates data is not missing.
  """

  def _retrieve_data_and_mask( self, ip ):
    if isinstance(ip, tuple):
      data, mask = ip
    elif isinstance(ip, dict):
      data, mask = ip["data"], ip["mask"]
    else:
      data, mask = ip, None
    return data, mask

  @tf.function
  def _mask_multiply(self, val, mask):
    if mask is None:
      return val
    return tf.multiply(val, mask)

  @tf.function
  def _anti_mask(self, mask, ):
    if mask is None:
      return None
    ret = tf.subtract( tf.constant(1.,dtype=tf.float32), mask )
    return ret

  @tf.function
  def _anti_mask_multiply(self, val, mask):
    if mask is None:
      return val
    return tf.multiply(val, self._anti_mask(mask) )

  @tf.function
  def _compose(self, orig, star, mask):
    if mask is None:
      return orig
    # NOTE: Avoid non-differentiable operations, such as
    #tf.where(tf.equal(mask,tf.constant(1.,dtype=tf.float32)), orig, star)
    pos = self._mask_multiply(orig, mask)
    anti = self._anti_mask_multiply(star, mask,)
    return tf.add( pos,  anti )

  @tf.function
  def _reduce_mean_mask( self, tensor, mask ):
    if mask is None:
      return tf.reduce_mean( tensor, axis = [0,1], keepdims = False)
    return tf.squeeze( 
      self._reduce_mean_mask_per_example(
          self._reduce_mean_mask_per_feature( tensor, mask )
        , mask )
      , axis = [0,1]
    )

  @tf.function
  def _valid_examples( self, mask, keepdims = True, cast_reduce = True ):
    if mask is None:
      return None
    if cast_reduce:
      mask = tf.cast( tf.reduce_any( tf.cast( mask, tf.bool ), axis = 1, keepdims = keepdims ) , tf.float32 )
    return tf.reduce_sum( mask, axis = 0, keepdims = keepdims )

  @tf.function
  def _valid_features( self, mask, keepdims = True ):
    if mask is None:
      return None
    return tf.reduce_sum( mask, axis = 1, keepdims = keepdims )

  @tf.function
  def _reduce_mean_mask_per_feature( self, tensor, mask ):
    if mask is None:
      return tf.reduce_mean( tensor, axis = 1, keepdims = True)
    # NOTE Reduce mean mask is important to ensure that all examples input the
    # same average contribution to the gradient
    # TODO Check other divides in code
    return tf.math.divide_no_nan(
        tf.reduce_sum(
            self._mask_multiply( tensor, mask )
        , axis = 1, keepdims = True )
        , self._valid_features( mask )
      )

  @tf.function
  def _reduce_mean_mask_per_example( self, tensor, mask ):
    if mask is None:
      return tf.reduce_mean( tensor, axis = 0, keepdims = True )
    # NOTE Reduce mean mask is important to ensure that all examples input the
    # same average contribution to the gradient
    # TODO Check other divides in code
    mask = tf.cast( tf.reduce_any( tf.cast( mask, tf.bool ), axis = 1, keepdims = True ) , tf.float32 )
    return tf.math.divide_no_nan(
      tf.reduce_sum(
          self._mask_multiply( tensor, mask )
      , axis = 0, keepdims = True )
      , self._valid_examples( mask, cast_reduce = False )
    ) 

  @tf.function
  def _numerically_stable_log(self, x):
    # NOTE https://stackoverflow.com/questions/33712178/tensorflow-nan-bug/42497444#42497444
    x_in_domain = tf.logical_and(
        tf.greater(x, 0.),
        tf.math.is_finite(x) )
    f = tf.math.log; safe_f = tf.zeros_like
    safe_x = tf.where(x_in_domain, x, tf.ones_like(x))
    return tf.where(x_in_domain, f(safe_x), safe_f(x))
