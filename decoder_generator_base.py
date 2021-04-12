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
    self._n_critic                  = retrieve_kw(kw, 'n_critic',                 0                            )
    self._result_file               = retrieve_kw(kw, 'result_file',              "weights.decoder_generator"  )
    self._latent_dim                = tf.constant( retrieve_kw(kw, 'latent_dim',  100                          ), dtype = tf.int32      )
    self._transform_to_meter_input  = retrieve_kw(kw, 'transform_to_meter_input', lambda x: x                  )
    self._n_pretrain_critic         = retrieve_kw(kw, 'n_pretrain_critic',        None                         )
    self._cache_performance_latent  = retrieve_kw(kw, 'cache_performance_latent', True                         )
    self._n_latent_performance_samples = retrieve_kw(kw, 'n_latent_performance_samples', 'performance_ds_cardinality' )
    super().__init__(data_sampler = data_sampler, **kw)
    # Define loss keys
    self._surrogate_lkeys |= {"critic_data", "critic_gen", "critic_total"}
    self._model_io_keys |= {"generator","critic",}
    # Define optimizers
    self._gen_opt    = self._add_optimizer( "generator", retrieve_kw(kw, 'gen_opt',  tf.optimizers.RMSprop(lr=1e-4, rho=.5) ) )
    self._critic_opt = self._add_optimizer( "critic", retrieve_kw(kw, 'critic_opt',  tf.optimizers.RMSprop(lr=1e-4, rho=.5) ) )
    # Latent dataset for performance evaluation
    self._cached_filepath_dict = {}
    self._decorate_latent_sampler()

  def performance_measure_fcn(self, sampler_ds, meters, lc):
    ret = {}
    # Loop over data samples
    for i, sample_batch in enumerate(sampler_ds):
      sample_batch = self.sample_parser_fcn( sample_batch )
      # TODO If needed, sample multiple batches per sample batch
      for meter in meters:
        meter.update_on_data_batch( sample_batch )
    # Loop over transported latent samples
    sample_iter = iter(sampler_ds)
    for latent_data in self._latent_sampler_performance_ds:
      sample_batch, sample_iter = self._secure_sample(sample_iter,sampler_ds)
      sample_batch = self.sample_parser_fcn( sample_batch )
      # FIXME This implementation can be improved
      # Retrieve generator equivalent data
      generator_batch = self.sample_generator_input(
            sampled_input = sample_batch
          , latent_data = latent_data
      )
      gen_batch = self.transform( generator_batch )
      for meter in filter(lambda m: isinstance(m, GenerativeEffMeter),meters):
        meter.update_on_gen_batch( gen_batch )
    # Retrieve results
    for meter in meters:
      ret.update(meter.retrieve())
      meter.reset()
    return ret

  def _secure_sample(self, sample_iter, sampler_ds):
    try:
      return next(sample_iter), sample_iter
    except StopIteration:
      sample_iter = iter(sampler_ds)
      return next(sample_iter), sample_iter

  def latent_sampler_ds_factory(self, opts, cache_filepath = ''):
    def infinite_generator():
      while True:
        yield self.sample_latent(opts.batch_size)
    ds = tf.data.Dataset.from_generator(
          infinite_generator
        , output_signature=tf.TensorSpec(shape=(opts.batch_size,self._latent_dim), dtype=tf.float32)
    ) # infinite loop dataset
    if cache_filepath: cache_filepath += '_batch%d' % opts.batch_size
    if bool(opts.take_n): # 
      if cache_filepath: cache_filepath += '_take%d' % opts.take_n
      from math import ceil
      ds = ds.take( int(ceil(opts.take_n / opts.batch_size)) )
    if cache_filepath:
      if cache_filepath not in self._cached_filepath_dict:
        mkdir_p(cache_filepath)
        ds = ds.cache( cache_filepath )
        self._cached_filepath_dict[cache_filepath] = ds
      else:
        ds = ds.cache()
        print("Warning: Caching on memory although specified to cache on disk.\nReason: Dataset at '%s' is already currently being cached." % cache_filepath )
    if opts.memory_cache:
      ds = ds.cache()
    if not os.path.exists(cache_filepath):
      # Force cache latent data
      for _ in ds: pass
    return ds

  def sample_generator_input(self, sampled_input = None, latent_data = None, n_samples = None, ds = None):
    if n_samples is None:
      n_samples = self.data_sampler.training_sampler_opts.batch_size if sampled_input is None else tf.shape(sampled_input)[0]
    if latent_data is None:
      return self.sample_latent(n_samples)
    else:
      return latent_data[:n_samples,...]

  def extract_condition_from_data(self, data ):
    return None

  def extract_target_space_from_data(self, data ):
    return data

  def build_input(self, condition, data):
    return data

  def extract_condition_from_generator_input(self, data ):
    # Assume that condition on generator input has the same position as in data
    return self.extract_condition_from_data( data )

  def extract_latent_from_generator_input(self, data ):
    return data

  def build_generator_input(self, condition, latent):
    return latent

  def sample_latent(self, n_samples):
    raise NotImplementedError("Overload this method with a latent sampling method")

  def _build_critic(self):
    raise NotImplementedError("Overload this method with a critic model")

  def _build_generator(self):
    raise NotImplementedError("Overload this method with a generator model")

  def _train_base(self, epoch, step, sample_batch):
    if (self._n_critic and (step % self._n_critic)) or (step != 0 and self._n_pretrain_critic is not None and step < self._n_pretrain_critic):
      # Update only critic
      loss_dict = self._train_step( samples = sample_batch, critic_only = True)
    else:
      # Update critic and generator
      loss_dict = self._train_step( samples = sample_batch, critic_only = False)
    return loss_dict

  def _decorate_latent_sampler(self):
    # TODO If adding a multiple of latent samples per sample batch, multiuple it on take_n 
    def get_cardinality(ds):
      card = ds.cardinality()
      if card == tf.data.UNKNOWN_CARDINALITY:
        for card, _ in enumerate(ds): pass
        card += 1
      return card
    if self._n_latent_performance_samples == "performance_ds_cardinality":
      card_train = get_cardinality(self.data_sampler.evaluation_sampler_from_train_ds)
      print("Train dataset cardinality is %s." % card_train)
      card_val = get_cardinality(self.data_sampler.evaluation_sampler_from_val_ds)
      print("Validation dataset cardinality is %s." % card_val)
      card = max(card_train, card_val)
      take_n = card
    else:
      take_n = self._n_latent_performance_samples
    from copy import copy
    opts = copy(self.data_sampler.eval_sampler_opts)
    opts.take_n = take_n
    latent_cache_path = self.data_sampler._cache_filepath + "_latent_data" if self._cache_performance_latent else ''
    self._latent_sampler_performance_ds = self.latent_sampler_ds_factory( opts, latent_cache_path )

