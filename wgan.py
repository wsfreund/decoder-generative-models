import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

try:
  from misc import *
  from eff_meter import *
  from decoder_generator_base import DecoderGenerator
except ImportError:
  from .misc import *
  from .eff_meter import *
  from .decoder_generator_base import DecoderGenerator

class Wasserstein_GAN(DecoderGenerator):

  def __init__(self, data_sampler, **kw):
    super().__init__(data_sampler, **kw)
    self._tf_call_kw           = retrieve_kw(kw, 'tf_call_kw',           {}                                                         )
    self._use_lipschitz_penalty = retrieve_kw(kw, 'use_gradient_penalty', True                                                      )
    self._grad_weight          = tf.constant( retrieve_kw(kw, 'grad_weight',          10.0                                        ) )
    self._surrogate_lkeys |= {"lipschitz", "wasserstein"}
    self._train_perf_lkeys |= set(["wasserstein",]) if [m for m in self._train_perf_meters if isinstance(m, CriticEffMeter)] else set()
    self._val_perf_lkeys |= set(["wasserstein",]) if [m for m in self._val_perf_meters if isinstance(m, CriticEffMeter)] else set()

  def latent_dim(self):
    return self._latent_dim

  @tf.function
  def latent_log_prob(self, latent):
    prior = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(self._latent_dim),
                                                     scale_diag=tf.ones(self._latent_dim))
    return prior.log_prob(latent)

  @tf.function
  def wasserstein_loss(self, y_true, y_pred):
    critic_data = tf.reduce_mean(y_true)
    critic_generator = tf.reduce_mean(y_pred)
    return critic_data - critic_generator, critic_data, critic_generator

  @tf.function
  def sample_latent(self, n_samples):
    return tf.random.normal((n_samples, self._latent_dim))

  @tf.function
  def transform(self, latent, **call_kw):
    return self.generator( latent, **call_kw)

  @tf.function
  def generate(self, n_samples, **call_kw):
    return self.transform( self.sample_generator_input( n_samples ),**call_kw)

  @tf.function
  def _compute_u_hat(self, x, x_hat):
    epsilon = tf.random.uniform((x.shape[0], 1, 1), 0.0, 1.0)
    u_hat = epsilon * x + (1 - epsilon) * x_hat
    return u_hat

  @tf.function
  def _lipschitz_penalty(self, x, x_hat):
    u_hat = self._compute_u_hat(x, x_hat)
    with tf.GradientTape() as penalty_tape:
      penalty_tape.watch(u_hat)
      func = self.critic(u_hat, **self._training_kw)
    grads = penalty_tape.gradient(func, u_hat)
    norm_grads = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
    regularizer = tf.math.square( tf.reduce_mean((norm_grads - 1) ) )
    return regularizer

  @tf.function
  def _surrogate_loss( self, data_output, gen_output, critic_lipschitz ):
    wasserstein_loss, critic_data, critic_generator = self.wasserstein_loss(data_output, gen_output)
    critic_total = tf.add( wasserstein_loss, critic_lipschitz )
    return { "critic_total":     critic_total
           , "lipschitz":        critic_lipschitz
           , "critic_data":      critic_data
           , "critic_gen":       critic_generator
           , "wasserstein":      wasserstein_loss }

  def _apply_critic_update( self, critic_tape, critic_loss ):
    critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
    if self._use_grad_clipping:
      critic_grads = [self._grad_clipping_fcn(g) for g in critic_grads if g is not None]
    self._critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
    return

  def _apply_gen_update( self, gen_tape, gen_loss):
    gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
    if self._use_grad_clipping:
      gen_grads = [self._grad_clipping_fcn(g) for g in gen_grads if g is not None]
    self._gen_opt.apply_gradients(zip(gen_grads, self.generator.trainable_variables))
    return

  @tf.function
  def _train_step(self, samples, fake_cond_samp = None, critic_only = False):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as critic_tape:
      gen_samples = self.generate( self.data_sampler._batch_size, **self._training_kw )
      data_output = self.critic(samples, **self._training_kw)
      gen_output = self.critic(gen_samples, **self._training_kw)
      lipschitz = self._lipschitz_penalty(samples, gen_samples) if self._use_lipschitz_penalty else tf.constant(0.)
      surrogate_loss_dict = self._surrogate_loss( data_output, gen_output, lipschitz )
    # gen_tape, critic_tape
    self._apply_critic_update( critic_tape, surrogate_loss_dict["critic_total"] )
    if not critic_only:
      self._apply_gen_update( gen_tape, surrogate_loss_dict["critic_gen"] )
    return surrogate_loss_dict
