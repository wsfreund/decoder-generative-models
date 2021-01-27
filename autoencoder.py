import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

try:
  from misc import *
  from train_base import TrainBase
except ImportError:
  from .misc import *
  from .train_base import TrainBase

class AutoEncoder( TrainBase ):
  """
  Custom surrogate training functions can be implemented by overloading

  self._compute_surrogate_loss( self, x, x_reco ):

  In order to compute other performance measures, overload

  self._performance_measure_fcn(sampler_ds)
  """

  def __init__(self, **kw):
    if not hasattr(self,'_required_models'):
      self._required_models = {"encoder", "decoder", "reconstructor"}
    else:
      self._required_models |= {"encoder", "decoder", "reconstructor"}
    super().__init__( **kw )
    # Retrieve optimizer
    self._ae_opt = retrieve_kw(kw, 'ae_opt', tf.optimizers.Adam() ) 
    # Define loss keys
    self._surrogate_lkeys |= {"ae_total"}
    self._train_perf_lkeys |= {"ae_total"}
    self._val_perf_lkeys |= {"ae_total"}
    # Overwrite default early_stopping_key
    self.early_stopping_key = retrieve_kw(kw, 'early_stopping_key', 'ae_total' )
    self._optimizer_dict.update({ "reconstructor" : self._ae_opt })
    self._model_io_keys |= {"encoder","decoder","reconstructor"}

  @tf.function
  def encode(self, x, **call_kw):
    return self.encoder( x, **call_kw )

  @tf.function
  def decode(self, code, **call_kw):
    return self.decoder( code, **call_kw )

  @tf.function
  def reconstruct(self, x, **call_kw):
    return self.reconstructor( x, **call_kw )

  def performance_measure_fcn(self, sampler_ds, meters, lc):
    # FIXME Currently meters_dict is being ignored. 
    # TODO Probably this function can be implemented only on train_base
    # and use self._compute_target instead of self.reconstruct to
    # feed meter.accumulate( data, output, target) batches.
    #import datetime
    #start = datetime.datetime.now()
    #print("Measuring performance...")
    # loss_fcn, prefix
    final_loss_dict = {}
    total_valid_examples = 0
    for sample_batch in sampler_ds:
      if self.sample_parser_fcn is not None:
        sample_batch = self.sample_parser_fcn( sample_batch )
      outputs = self.reconstruct( sample_batch )
      ## compute loss
      reco_loss_dict = self._compute_surrogate_loss( sample_batch, outputs )
      data, mask = self._retrieve_data_and_mask( sample_batch )
      ## valid examples
      valid_examples = data.shape[0] if mask is None else self._valid_examples( mask, keepdims = False )
      total_valid_examples += valid_examples # keep track of total number of valid samples
      reco_loss_dict["ae_total"] *= valid_examples # denormalize
      ## accumulate
      self._accumulate_loss_dict( final_loss_dict, reco_loss_dict )
    ## Renormalize
    final_loss_dict["ae_total"] /= ( total_valid_examples if total_valid_examples else 1 )
    final_loss_dict = self._parse_surrogate_loss( final_loss_dict )
    #total_time = datetime.datetime.now() - start
    #print("Finished measuring performance in %s." % total_time)
    return final_loss_dict

  @tf.function
  def _compute_surrogate_loss( self, x, x_reco ):
    x, mask = self._retrieve_data_and_mask( x )
    reco_numerical = self._reduce_mean_mask( 
      tf.square( 
        tf.subtract( x, x_reco )
      )
    , mask ) if mask is None or tf.reduce_any(tf.cast(mask, tf.bool)) else tf.constant(0., dtype=tf.float32)
    return { 'ae_total' : reco_numerical }

  #@tf.function
  def _apply_ae_update( self, ae_tape, ae_loss):
    ae_variables = self.reconstructor.trainable_variables
    ae_grads = ae_tape.gradient(ae_loss, ae_variables)
    if self._use_grad_clipping:
      ae_grads = [self._grad_clipping_fcn(g) for g in ae_grads if g is not None]
    self._ae_opt.apply_gradients(zip(ae_grads, ae_variables))
    return

  @tf.function
  def _train_step(self, x ):
    with tf.GradientTape() as ae_tape:
      x_reco = self.reconstruct( x, **self._training_kw )
      ae_loss_dict = self._compute_surrogate_loss( x, x_reco )
    # ae_tape,
    self._apply_ae_update( ae_tape, ae_loss_dict['ae_total'] )
    return ae_loss_dict
