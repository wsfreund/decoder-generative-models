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
    self.conditioning_sampler             = retrieve_kw(kw, 'conditioning_sampler',  None  )
    self._use_same_real_fake_conditioning = retrieve_kw(kw, 'use_same_real_fake_conditioning',  False )
    self._extract_conditioning_fcn        = retrieve_kw(kw, 'extract_conditioning_fcn', None )
    if self.conditioning_sampler is None and not(self._use_same_real_fake_conditioning):
      raise ValueError("Specify conditioning_sampler when setting use_same_real_fake_conditioning to False")
    if not(self._use_same_real_fake_conditioning):
      self._cached_train_cond_sampler = iter(self.conditioning_sampler.sampler_from_train_ds)
    elif self._extract_conditioning_fcn is None:
      raise ValueError("Specify how to extract the conditioning from the samples with the conditioning_sampler when setting use_same_real_fake_conditioning to True")

  def _sample_conditioning_batch(self, sample_batch):
    if self._use_same_real_fake_conditioning:
      additional_samples = sample_batch
      additional_samples = self._extract_conditioning_fcn( additional_samples )
    else:
      try:
        additional_samples = next(self._cached_train_cond_sampler)
      except StopIteration:
        self._cached_train_cond_sampler = iter(self.conditioning_sampler.sampler_from_train_ds)
        additional_samples = next(self._cached_train_cond_sampler)
      additional_samples = self._sample_parser_fcn( additional_samples )
    return additional_samples

  def _train_base(self, epoch, step, sample_batch):
    additional_samples = self._sample_conditioning_batch(sample_batch)
    if self._n_critic and (step % self._n_critic):
      # Update only critic
      loss_dict = self._train_step(sample_batch, additional_samples = additional_samples, critic_only = True)
    if not(self._n_critic) or not(step % self._n_critic):
      # Update critic and generator
      loss_dict = self._train_step(sample_batch, additional_samples = additional_samples, critic_only = False)
    return loss_dict
