import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

try:
  from misc import *
  from train_base import TrainBase
  from embedding_base import ModelWithEmbeddings
except ImportError:
  from .misc import *
  from .train_base import TrainBase
  from .embedding_base import ModelWithEmbeddings

class AutoEncoderWithEmbeddings( ModelWithEmbeddings, AutoEncoder ):

  def __init__(self, input_info_dict, **kw):
    super().__init__(input_info_dict = input_info_dict, **kw )
    # Define loss keys
    self._surrogate_lkeys  |= {"ae_total", "ae_numerical", "ae_categorical", "ae_sigmoid", "ae_softmax"}
    self._required_models |= {"train_model"} # TODO
    # Define loss function for performance evaluation (not training)

  @tf.function
  def train_reconstruct(self, x ):
    return self.train_model( x, **self._training_kw )

  def performance_measure_fcn(self, sampler_ds):
    # loss_fcn, prefix
    final_loss_dict = {}
    loss_keys = ("ae_numerical", "ae_softmax", "ae_sigmoid")
    mask_fcns = (self._compute_numerical_mask, self._compute_softmax_mask, self._compute_sigmoid_mask )
    total_valid_examples = { k : 0 for k in loss_keys }
    for sample_batch in sampler_ds:
      if self.sample_parser_fcn is not None:
        sample_batch = self.sample_parser_fcn( sample_batch )
      outputs = self.train_model( sample_batch )
      ## compute loss
      reco_loss_dict = self._compute_surrogate_loss( sample_batch, outputs )
      data, mask = self._retrieve_data_and_mask( sample_batch )
      ## valid examples
      for loss_key, mask_fcn in zip( loss_keys, mask_fcns ):
        valid_examples = data.shape[0] if mask is None else self._valid_examples( mask_fcn( mask ), keepdims = False )
        total_valid_examples[loss_key] += valid_examples # keep track of total samples
        reco_loss_dict[loss_key] *= valid_examples # denormalize
      ## accumulate
      self._accumulate_loss_dict( final_loss_dict, reco_loss_dict )
    ## Renormalize
    for loss_key in loss_keys:
      final_loss_dict[loss_key] /= ( total_valid_examples[loss_key] if total_valid_examples[ loss_key ] else 1 )
    final_loss_dict["ae_categorical"] = final_loss_dict[ "ae_sigmoid"] + final_loss_dict[ "ae_softmax"]
    final_loss_dict["ae_total"]       = final_loss_dict[ "ae_categorical"] + final_loss_dict[ "ae_numerical"]
    final_loss_dict = self._parse_surrogate_loss( final_loss_dict, prefix = prefix )
    return final_loss_dict

  @tf.function
  def _compute_surrogate_loss( self, x, outputs ):
    if self._has_softmax:
      softmax_inputs    = outputs[self._softmax_input_slice]
      softmax_logits    = outputs[self._softmax_logits_slice]
      ae_softmax        = self._compute_softmax_loss( softmax_inputs, softmax_logits, mask )
    else:
      ae_softmax      = tf.constant(0., dtype=tf.float32 )
    if self._has_sigmoid:
      sigmoid_inputs    = outputs[self._sigmoid_input_slice]
      sigmoid_logits    = outputs[self._sigmoid_logits_slice]
      ae_sigmoid        = self._compute_sigmoid_loss( sigmoid_inputs, sigmoid_logits, mask )
    else:
      ae_sigmoid     = tf.constant(0., dtype=tf.float32 )
    if self._has_numerical:
      numerical_inputs  = outputs[self._numerical_input_slice][0]
      numerical_outputs = outputs[self._numerical_output_slice][0]
      ae_numerical    = self._compute_numerical_loss( numerical_inputs, numerical_outputs, mask )
    else:
      ae_numerical    = tf.constant(0., dtype=tf.float32 )
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
      x_reco = self.train_reconstruct( x )
      ae_loss_dict = self._compute_surrogate_loss( x, x_reco )
    # ae_tape,
    self._apply_ae_update( ae_tape, ae_loss_dict['ae_total'] )
    return ae_loss_dict
