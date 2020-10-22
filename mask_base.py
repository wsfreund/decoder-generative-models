import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

try:
  from misc import *
except ImportError:
  from .misc import *

class MaskModel(object):

  @tf.function
  def _mask_multiply(self, val, mask):
    return tf.multiply(val, mask)

  @tf.function
  def _anti_mask(self, mask, ):
    ret = tf.subtract( tf.constant(1.,dtype=tf.float32), mask )
    return ret

  @tf.function
  def _anti_mask_multiply(self, val, mask):
     return tf.multiply(val, self._anti_mask(mask) )

  @tf.function
  def _compose(self, orig, star, mask):
    # NOTE: Avoid non-differentiable operations, such as
    #tf.where(tf.equal(mask,tf.constant(1.,dtype=tf.float32)), orig, star)
    pos = self._mask_multiply(orig, mask)
    anti = self._anti_mask_multiply(star, mask,)
    return tf.add( pos,  anti )

  @tf.function
  def _reduce_mean_mask( self, tensor, mask ):
    return tf.squeeze( 
      self._reduce_mean_mask_per_example(
          self._reduce_mean_mask_per_feature( tensor, mask )
        , mask )
      , axis = [0,1]
    )

  @tf.function
  def _valid_examples( self, mask, keepdims = True, cast_reduce = True ):
    if cast_reduce:
      mask = tf.cast( tf.reduce_any( tf.cast( mask, tf.bool ), axis = 1, keepdims = keepdims ) , tf.float32 )
    return tf.reduce_sum( mask, axis = 0, keepdims = keepdims )

  @tf.function
  def _valid_features( self, mask, keepdims = True ):
    return tf.reduce_sum( mask, axis = 1, keepdims = keepdims )

  @tf.function
  def _reduce_mean_mask_per_feature( self, tensor, mask ):
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

  def _create_mask_from_slice(self, s):
    mask = np.zeros((1, self._n_mask_inputs), dtype=np.bool)
    for i in s:
      mask[:,i] = True
    return tf.constant(mask, dtype=tf.bool), [len(s),]
