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
  """
  NOTE: Current code requires that the generator output has the same format as
  the discriminator input, including any conditioning. I.e. the generator must
  output the same conditions it used as input, to bypass it to the critic.
  """
  def __init__(self, data_sampler, **kw):
    super().__init__( data_sampler, **kw )
    # When set to True, must specify 
    self.use_same_real_fake_conditioning = retrieve_kw(kw, 'use_same_real_fake_conditioning',  True )
    if self.use_same_real_fake_conditioning is False:
      raise NotImplementedError("use_same_real_fake_conditioning must be set to True.")
    # This must be specified to allow comparing real with fake samples on
    # the "self.generate" function and to sample when using
    # use_same_real_fake_conditioning
    self.extract_generator_input_from_standard_batch_fcn = retrieve_kw(kw, 'extract_generator_input_from_standard_batch_fcn', None )
    # Supposed to sample all info except the latent space
    self.generator_sampler = retrieve_kw(kw, 'generator_sampler',  None  )

  def sample_generator_input(self, sampled_input = None, latent_data = None, n_samples = None, ds = None):
    if self.use_same_real_fake_conditioning:
      sampler = self.data_sampler
    else:
      # FIXME note that if use_same_real_fake_conditioning is set to false, the
      # generator_sampler is not used when sampled_input is specified.
      sampler = self.generator_sampler
    if sampled_input is None:
      if ds is None:
        raise ValueError("ds must be specified if not specifying sampled_input.")
      def safe_sampler( iter_prop, sampler ):
        try:
          sampled_input = next(self.__dict__[iter_prop])
        except (StopIteration, KeyError):
          self.__dict__[iter_prop] = iter(sampler)
          sampled_input = next(self.__dict__[iter_prop])
        return sampled_input
      if n_samples is None:
        if ds == 'train':
          sampled_input = safe_sampler("_cached_train_sampler", sampler.evaluation_sampler_from_train_ds)
        elif ds == 'val':
          sampled_input = safe_sampler("_cached_val_sampler", sampler.evaluation_sampler_from_val_ds)
        elif ds == 'test':
          sampled_input = safe_sampler("_cached_test_sampler", sampler.evaluation_sampler_from_test_ds)
      else:
        sampled_input = sampler.sample( n_samples = n_samples, ds = ds )
    if self.use_same_real_fake_conditioning:
      if self.extract_generator_input_from_standard_batch_fcn is None:
        raise NotImplementedError(
            """Please specify extract_generator_input_from_standard_batch_fcn
            with a method capable of extracting all required information from
            regular batch except for the latent states."""
        )
      sampled_input = self.extract_generator_input_from_standard_batch_fcn( sampled_input )
    if not isinstance(sampled_input, list):
      sampled_input = [sampled_input]
    if n_samples is None:
      n_samples = sampled_input[0].shape[0]
    if latent_data is None:
      latent_data = self.sample_latent( n_samples )
    else:
      if latent_data.shape[0] != n_samples:
        raise ValueError("latent_data size differs from number of samples")
    generator_input = sampled_input + [ latent_data ]
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
    # FIXME Ensure that sample generator uses generator conditioning
    generator_batch = self.sample_generator_input( sample_batch, ds = "train" )
    if (self._n_critic and (step % self._n_critic)) or (step != 0 and self._n_pretrain_critic is not None and step < self._n_pretrain_critic):
      # Update only critic
      loss_dict = self._train_step(data_batch = sample_batch, gen_batch = generator_batch, critic_only = True)
    else:
      # Update critic and generator
      loss_dict = self._train_step(data_batch = sample_batch, gen_batch = generator_batch, critic_only = False)
    return loss_dict
