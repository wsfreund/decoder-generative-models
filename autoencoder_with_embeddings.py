import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

try:
  from misc            import *
  from train_base      import TrainBase
  from autoencoder     import AutoEncoder
  from embedding_base  import ModelWithEmbeddings
except ImportError:
  from .misc           import *
  from .train_base     import TrainBase
  from .autoencoder    import AutoEncoder
  from .embedding_base import ModelWithEmbeddings

class AutoEncoderWithEmbeddings( ModelWithEmbeddings, AutoEncoder ):

  def __init__(self, data_sampler, **kw):
    super().__init__(data_sampler = data_sampler, **kw)
    self._surrogate_lkeys  |= {"ae_total", "ae_numerical", "ae_categorical", "ae_sigmoid", "ae_softmax"}

  @tf.function
  def _compute_surrogate_loss( self, x, outputs ):
    if self._has_softmax:
      softmax_targets   = outputs['softmax_targets']
      softmax_logits    = outputs['softmax_logits']
      ae_softmax        = self._compute_softmax_loss( softmax_targets, softmax_logits )
    else:
      ae_softmax        = tf.constant(0., dtype=tf.float32 )
    if self._has_sigmoid:
      sigmoid_targets   = outputs['sigmoid_targets']
      sigmoid_logits    = outputs['sigmoid_logits']
      ae_sigmoid        = self._compute_sigmoid_loss( sigmoid_targets, sigmoid_logits )
    else:
      ae_sigmoid        = tf.constant(0., dtype=tf.float32 )
    if self._has_numerical:
      numerical_targets = outputs['numerical_targets']
      numerical_outputs = outputs['numerical_outputs']
      ae_numerical      = self._compute_numerical_loss( numerical_targets, numerical_outputs )
    else:
      ae_numerical      = tf.constant(0., dtype=tf.float32 )
    ae_categorical    = tf.add( ae_softmax, ae_sigmoid )
    ae_total = tf.add( ae_numerical, ae_categorical )
    return { 'ae_total' :          ae_total
           , 'ae_numerical' :      ae_numerical
           , 'ae_categorical' :    ae_categorical
           , 'ae_sigmoid' :        ae_sigmoid 
           , 'ae_softmax' :        ae_softmax
           }

  @tf.function
  def _train_step(self, x ):
    with tf.GradientTape() as ae_tape:
      x_training_reco = self._training_model( x, **self._training_kw )
      ae_loss_dict = self._compute_surrogate_loss( x, x_training_reco )
    # ae_tape,
    self._apply_ae_update( ae_tape, ae_loss_dict['ae_total'] )
    return ae_loss_dict
