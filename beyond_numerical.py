import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

try:
  from misc import *
  from mask_base import MaskModel
  from train_base import TrainBase
except ImportError:
  from .misc import *
  from .mask_base import MaskModel
  from .train_base import TrainBase

class InputInfo(object):
  def __init__( self, variable_names
              , variable_indices ):
    self.variable_names = variable_names
    self.variable_indices = variable_indices

  @property
  def n_variables( self ):
    return len(self.variable_names)

  def __repr__( self ):
    return ( self.__class__.__name__ + "(" + str(self.variable_indices) + "|" + str(self.n_variables) + ")" )

class NumericalInputInfo(InputInfo):
  def __init__( self
              , variable_names
              , variable_indices ):
    InputInfo.__init__(self
        , variable_names             = variable_names
        , variable_indices             = variable_indices )

  @property
  def categorical_mask( self ):
    return [0]*self.n_variables

  @property
  def numerical_mask( self ):
    return [1]*self.n_variables

class CategoricalInputInfoBase(InputInfo):
  def __init__( self
      , category_name
      , variable_indices
      , variable_names ):
    self.category_name = category_name
    InputInfo.__init__(self
        , variable_names             = variable_names
        , variable_indices             = variable_indices )

  @property
  def categorical_mask( self ):
    return [1]*self.n_variables

  @property
  def numerical_mask( self ):
    return [0]*self.n_variables

  def __repr__( self ):
    return ( self.__class__.__name__ + "(" +  str(self.variable_indices) + "|var:" + self.category_name + "|"
        + str(self.n_variables) + ")" )


class BinaryInputInfo(CategoricalInputInfoBase):
  def __init__( self
      , category_name
      , variable_indices
      , variable_names ):
    self.category_name = category_name
    InputInfo.__init__(self
        , variable_names             = variable_names
        , variable_indices             = variable_indices )
    assert len(self.variable_names) == 2

  @property
  def n_variables( self ):
    return 1

class CategoricalGroupInputInfo(CategoricalInputInfoBase):
  def __init__( self
      , category_name
      , variable_indices
      , variable_names ):
    self.category_name = category_name
    InputInfo.__init__(self
        , variable_names             = variable_names
        , variable_indices             = variable_indices )
    assert len(self.variable_names) > 1

class BeyondNumericalDataModel(TrainBase):
  """
  Note: This class only works if batch dim is at tensor dimension 0
  """

  def __init__(self, data_sampler, **kw):
    super().__init__(data_sampler, **kw)
    #self._expand_mask_matrix = self._retrieve_mask_mat()

  @tf.function
  def _compute_numerical_loss( self, x, x_reco):
    x, mask = self._retrieve_data_and_mask( x )
    reco_numerical = self._reduce_mean_mask( 
      tf.square( 
        tf.subtract( x, x_reco )
      )
    , mask ) if mask is None or tf.reduce_any(tf.cast(mask, tf.bool)) else tf.constant(0., dtype=tf.float32)
    return reco_numerical

  @tf.function
  def _compute_sigmoid_loss( self, labels, logits):
    labels, mask = self._retrieve_data_and_mask( labels )
    if mask is not None:
      if tf.math.logical_not(tf.reduce_any(tf.cast(mask, tf.bool))):
        return tf.constant(0., dtype=tf.float32)
    loss = []; count = 0;
    for i, (label, logit) in enumerate(zip(labels, logits)):
      per_example_loss = tf.expand_dims( tf.squeeze( tf.nn.sigmoid_cross_entropy_with_logits(
            labels = label,
            logits = logit ) ), axis = 1 )
      m = tf.expand_dims( mask[:,i], axis = 1 ) if mask is not None else None
      category_loss = tf.squeeze( 
          self._reduce_mean_mask_per_example( 
            per_example_loss 
          , m ) 
      )
      loss.append( category_loss )
      count += tf.cast( tf.reduce_any(tf.cast(m, tf.bool)) if mask is not None else tf.shape( label )[0], tf.float32 )
    tot = tf.math.divide_no_nan( tf.reduce_sum( loss ), count )
    return tot

  @tf.function
  def _compute_softmax_loss( self, labels, logits):
    labels, mask = self._retrieve_data_and_mask( labels )
    if mask is not None:
      if tf.math.logical_not(tf.reduce_any(tf.cast(mask, tf.bool))):
        return tf.constant(0., dtype=tf.float32)
    loss = []; count = 0;
    for i, (label, logit) in enumerate(zip(labels, logits)):
      per_example_loss = tf.expand_dims( tf.nn.softmax_cross_entropy_with_logits(
            labels = label,
            logits = logit ), axis = 1 )
      m = tf.expand_dims( mask[:,i], axis = 1 ) if mask is not None else None
      category_loss = tf.squeeze( 
          self._reduce_mean_mask_per_example( 
            per_example_loss 
          , m ) 
      )
      loss.append( category_loss )
      count += tf.cast( tf.reduce_any(tf.cast(m, tf.bool)) if mask is not None else tf.shape( label )[0], tf.float32 ) 
    tot = tf.math.divide_no_nan( tf.reduce_sum( loss ), count )
    return tot

  def _parse_surrogate_loss(self, train_loss, prefix = ''):
    # TODO Make a set of prefix
    if prefix and not(prefix.endswith('_')): prefix += '_'
    train_loss = TrainBase._parse_surrogate_loss(self, train_loss)
    if (prefix + 'numerical') in train_loss:
      train_loss[prefix + 'numerical'] = np.sqrt(train_loss[prefix + 'numerical'])
      train_loss[prefix + 'total'] = train_loss[prefix + 'numerical'] + train_loss[prefix + 'categorical']
    return train_loss

  #def _retrieve_mask_mat( self ):
  #  # TODO
  #  mat = np.zeros((self._n_mask_inputs,self._n_features), dtype=np.float32)
  #  l = 0; c = 0
  #  for info in self._input_info_dict.values(): # TODO
  #    n=info.n_variables
  #    if isinstance(info, CategoricalInputInfoBase):
  #      mat[l,c:(c+n)] = 1.
  #      l+=1
  #    else:
  #      for l2 in range(n):
  #        mat[l+l2,c+l2] = 1.
  #      l+=l2
  #    c+=n
  #  return tf.constant( mat, dtype=tf.float32 )

  #@tf.function
  #def _expand_mask( self, mask ):
  #  return tf.linalg.matmul( mask, self._expand_mask_matrix )

