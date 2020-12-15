import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tempfile

try:
  from misc import *
  from train_base import TrainBase
except ImportError:
  from .misc import *
  from .train_base import TrainBase


class DecoderGenerator(TrainBase):
  def __init__(self, data_sampler, **kw):
    super().__init__(data_sampler = data_sampler, **kw)
    self._n_critic             = retrieve_kw(kw, 'n_critic',               0                          )
    self._result_file          = retrieve_kw(kw, 'result_file',  "weights.decoder_generator"          )
    self._latent_dim           = tf.constant( retrieve_kw(kw, 'latent_dim', 100 ), dtype = tf.int64   ) 
    self._gen_opt              = retrieve_kw(kw, 'gen_opt',              tf.optimizers.Adam()         )
    self._critic_opt           = retrieve_kw(kw, 'critic_opt',           tf.optimizers.Adam()         )
    self._n_pretrain_critic    = retrieve_kw(kw, 'n_pretrain_critic',    None                         )
    self._lkeys |= {"critic", "generator"}
    self._val_lkeys |= {"generator", "step"}
    # build models
    self.critic = self._build_critic()
    self.generator = self._build_generator()
    self._model_dict = { "generator" : self.generator
                       , "critic" : self.critic }
    self._optimizer_dict = { "generator" : self._gen_opt
                           , "critic" : self._critic_opt }
    if self._load_model_at_path:
      self.load( self._load_model_at_path, keys = ["generator","critic"] )

  def sample_generator_input(self, *args, **kwargs):
    raise NotImplementedError("Overload this method with a generator input samppling method")

  def sample_latent(self, nsamples):
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
