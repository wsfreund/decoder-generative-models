import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

try:
  from misc import *
  from train_base import TrainBase
  from eff_meter import *
except ImportError:
  from .misc import *
  from .train_base import TrainBase
  from .eff_meter import *

class AutoEncoder( TrainBase ):

  def __init__(self, data_sampler, **kw):
    if not hasattr(self,'_required_models'):
      self._required_models = {"encoder", "decoder", "reconstructor"}
    else:
      self._required_models |= {"encoder", "decoder", "reconstructor"}
    super().__init__(data_sampler = data_sampler, **kw)
    # Define optimizers
    self._ae_opt = self._add_optimizer( "reconstructor", retrieve_kw(kw, 'ae_opt', tf.optimizers.Adam() ) )
    self._add_optimizer( "encoder", None )
    self._add_optimizer( "decoder", None )
    # Define loss keys
    self._surrogate_lkeys |= {"ae_total"}
    # Overwrite default early_stopping_key
    self.early_stopping_key = retrieve_kw(kw, 'early_stopping_key', 'ae_total' )
    if not any(map(lambda x: isintance(m,AE_EffMeter), self._train_perf_meters)):
      meter = AE_EffMeter(); meter.initialize(self)
      self._train_perf_meters = [meter] + self._train_perf_meters
    if not any(map(lambda x: isintance(m,AE_EffMeter), self._val_perf_meters)):
      meter = AE_EffMeter(); meter.initialize(self)
      self._val_perf_meters = [meter] + self._val_perf_meters
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
    # FIXME Improve it to only allow not none for desired layers
    self._ae_opt.apply_gradients( (grad, var,) 
        for (grad, var) in zip(ae_grads, ae_variables) 
        if grad is not None
    )
    return

  @tf.function
  def _train_step(self, x ):
    with tf.GradientTape() as ae_tape:
      x_reco = self.reconstruct( x, **self._training_kw )
      ae_loss_dict = self._compute_surrogate_loss( x, x_reco )
    # ae_tape,
    self._apply_ae_update( ae_tape, ae_loss_dict['ae_total'] )
    return ae_loss_dict
