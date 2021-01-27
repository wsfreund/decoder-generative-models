import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import contextlib

try:
  from misc import *
  from train_base import TrainBase
  from eff_meter import *
except ImportError:
  from .misc import *
  from .train_base import TrainBase
  from .eff_meter import *

class DecoderGenerator(TrainBase):
  def __init__(self, data_sampler, **kw):
    if not hasattr(self,'_required_models'):
      self._required_models = {"generator", "critic",}
    else:
      self._required_models |= {"generator", "critic",}
    # TODO Set default optimizer to adam with zero momentum!
    self._n_critic                 = retrieve_kw(kw, 'n_critic',                 0                            )
    self._result_file              = retrieve_kw(kw, 'result_file',              "weights.decoder_generator"  )
    self._latent_dim  = tf.constant( retrieve_kw(kw, 'latent_dim',               100                          ), dtype = tf.int64      )
    self._gen_opt                  = retrieve_kw(kw, 'gen_opt',                  tf.optimizers.RMSprop(lr=1e-4, rho=.5)  )
    self._critic_opt               = retrieve_kw(kw, 'critic_opt',               tf.optimizers.RMSprop(lr=1e-4, rho=.5)  )
    self._transform_to_meter_input = retrieve_kw(kw, 'transform_to_meter_input', lambda x: x                  )
    self._n_pretrain_critic        = retrieve_kw(kw, 'n_pretrain_critic',        None                         )
    super().__init__(data_sampler = data_sampler, **kw)
    # Define loss keys
    self._surrogate_lkeys |= {"critic_data", "critic_gen", "critic_total"}
    self._train_perf_lkeys |= set(["critic_data", "critic_gen",]) if [m for m in self._train_perf_meters if isinstance(m, CriticEffMeter)] else set()
    self._val_perf_lkeys |= set(["critic_data", "critic_gen",]) if [m for m in self._val_perf_meters if isinstance(m, CriticEffMeter)] else set()
    # XXX
    if "critic" in self._surrogate_lkeys:  self._surrogate_lkeys.remove("critic")
    if "critic" in self._train_perf_lkeys: self._train_perf_lkeys.remove("critic")
    if "critic" in self._val_perf_lkeys:   self._val_perf_lkeys.remove("critic")
    # Define optimizers
    self._optimizer_dict.update({ 
        "generator" : self._gen_opt
      , "critic" : self._critic_opt
    })
    self._model_io_keys |= {"generator","critic",}
    for m in filter(lambda m: isinstance(m, CriticEffMeter), self._train_perf_meters):
      m.model = self._model_dict["critic"]
    for m in filter(lambda m: isinstance(m, CriticEffMeter), self._val_perf_meters):
      m.model = self._model_dict["critic"]

  def performance_measure_fcn(self, sampler_ds, meters, lc):
    gen_meters = list(filter(lambda m: isinstance(m,GenerativeEffMeter) and not isinstance(m,CriticEffMeter), meters))
    critic_meters = list(filter(lambda m: isinstance(m,CriticEffMeter), meters))
    # FIXME Currently only works for single batch
    final_loss_dict = {}
    if not gen_meters and not critic_meters:
      return final_loss_dict
    ## Prepare data
    sample_batch = self.sample_parser_fcn(next(iter(sampler_ds)))
    n_samples = sample_batch[0].shape[0]
    # TODO If we need different size of latent data, then the sampling of
    # latent data should be on the meters. Or think on another solution
    self._cache_performance_latent( n_samples )
    latent_data = self._performance_latent_data[:n_samples,...]
    #
    self._generator_batch = self.sample_generator_input(sample_batch
        , latent_data = latent_data
    )
    gen_batch = self.transform( self._generator_batch )
    # Make sure we are working on the correct format:
    data_meter_batch = self._transform_to_meter_input( sample_batch )
    gen_meter_batch = self._transform_to_meter_input( gen_batch )
    ## -- end of Prepare data
    def lrun( data, gen, lmeters ):
      for meter in lmeters:
        meter.initialize(data, gen)
      for meter in lmeters:
        meter.reset()
      # Accumulate gens
      for meter in lmeters:
        meter.accumulate(gen)
      # Retrieve efficiencies by computing summary statistics
      for meter in lmeters:
        # Keep track of results
        if isinstance(meter, CriticEffMeter): # XXX
          # Write on data summary
          critic_gen = meter.retrieve( gen = True )
          critic_data = meter.retrieve( gen = False )
          wasserstein = critic_data - critic_gen
          final_loss_dict[meter.name + "_gen"] = critic_gen
          final_loss_dict[meter.name + "_data"] = critic_data
          final_loss_dict["wasserstein"] = wasserstein
        else:
          final_loss_dict[meter.name] = meter.retrieve()
    # Run:
    lrun( data_meter_batch, gen_meter_batch, gen_meters)
    # XXX Find a better approach to CriticEffMeter
    if critic_meters:
      lrun( sample_batch, gen_batch, critic_meters)
    return final_loss_dict

  def sample_generator_input(self, sampled_input = None, latent_data = None, n_samples = None):
    if n_samples is None and latent_data is None:
      n_samples = self.data_sampler._batch_size if sampled_input is None else sampled_input.shape[0]
    if latent_data is None:
      return self.sample_latent(n_samples)
    else:
      return latent_data

  def sample_latent(self, n_samples):
    raise NotImplementedError("Overload this method with a latent sampling method")

  def _build_critic(self):
    raise NotImplementedError("Overload this method with a critic model")

  def _build_generator(self):
    raise NotImplementedError("Overload this method with a generator model")

  def _train_base(self, epoch, step, sample_batch):
    if (self._n_critic and (step % self._n_critic)) or (step != 0 and self._n_pretrain_critic is not None and step < self._n_pretrain_critic):
      # Update only critic
      loss_dict = self._train_step(sample_batch, critic_only = True)
    else:
      # Update critic and generator
      loss_dict = self._train_step(sample_batch, critic_only = False)
    return loss_dict

  def _cache_performance_latent(self, n_samples):
    if not hasattr(self,'_performance_latent_data') or n_samples > self._performance_latent_data.shape[0]:
      # TODO Grow caching tensor on demand
      # NOTE this approach makes _generator_batch shared between train/val datasets
      self._performance_latent_data = self.sample_latent( 
          n_samples = n_samples
      )

