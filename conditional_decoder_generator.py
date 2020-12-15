import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tempfile

try:
  from misc import *
  from decoder_generator_base import DecoderGenerator
except ImportError:
  from .misc import *
  from .decoder_generator_base import DecoderGenerator

class cDecoderGenerator(DecoderGenerator):
  def __init__(self, data_sampler, **kw):
    super().__init__( data_sampler, **kw )
    # When set to True, must specify 
    self.use_same_real_fake_conditioning = retrieve_kw(kw, 'use_same_real_fake_conditioning',  False )
    # This must be specified to allow comparing real with fake samples on
    # generate function and to sample when using
    # use_same_real_fake_conditioning
    self.extract_generator_input_from_standard_batch_fcn = retrieve_kw(kw, 'extract_generator_input_from_standard_batch_fcn', None )
    # Supposed to sample all info except the latent space
    self.generator_sampler = retrieve_kw(kw, 'generator_sampler',  None  )

  def sample_generator_input(self, sampled_input = None, n_samples = None, ds = 'trn'):
    if self.use_same_real_fake_conditioning:
      sampler = self.data_sampler
    else:
      sampler = self.generator_sampler
    def try_sampler( iter_prop, sampler ):
      try:
        sampled_input = next(self.__dict__[iter_prop])
      except (StopIteration, KeyError):
        self.__dict__ = iter(self.generator_sampler.sampler_from_train_ds)
        sampled_input = next(self._cached_train_sampler)
    if sampled_input is None:
      if n_samples is None:
        if ds == 'train':
          sampled_input = sample("_cached_train_sampler", sampler.sampler_from_train_ds)
        elif ds == 'val':
          sampled_input = sample("_cached_val_sampler", sampler.sampler_from_val_ds)
        elif ds == 'test':
          sampled_input = sample("_cached_test_sampler", sampler.sampler_from_test_ds)
      else:
        sampled_input = sampler.sample( n_samples = n_samples, ds = ds )
    if self.use_same_real_fake_conditioning:
      if self.extract_generator_input_from_standard_batch_fcn is None:
        raise NotImplementedError(
            """Please specify extract_generator_input_from_standard_batch_fcn
            with a method capable of extracting all required information from
            regular batch except for the latent states.""")
      sampled_input = self.extract_generator_input_from_standard_batch_fcn( sampled_input )
    if not isinstance(sampled_input, list):
      sampled_input = [sampled_input]
    if n_samples is None:
      n_samples = sampled_input[0].shape[0]
    generator_input = sampled_input + [ self.sample_latent( n_samples ) ]
    generator_input = self._ensure_batch_size_dim(self.generator, generator_input)
    return generator_input

  def _ensure_batch_size_dim(self, model, inputs):
    gm_len = len(model.input[0].shape)
    gi_len = len(inputs[0].shape)
    if gi_len > gm_len:
      raise ValueError("Extract generator input with size larger than actual model input")
    while gi_len != gm_len:
      inputs = [tf.expand_dims(i,axis=0) for i in inputs ]
      gi_len = len(inputs[0].shape)
    return inputs

  def _train_base(self, epoch, step, sample_batch):
    generator_batch = self.sample_generator_input(sample_batch)
    if (self._n_critic and (step % self._n_critic)) or (step != 0 and self._n_pretrain_critic is not None and step < self._n_pretrain_critic):
      # Update only critic
      loss_dict = self._train_step(real_batch = sample_batch, gen_batch = generator_batch, critic_only = True)
    else:
      # Update critic and generator
      loss_dict = self._train_step(real_batch = sample_batch, gen_batch = generator_batch, critic_only = False)
    return loss_dict
