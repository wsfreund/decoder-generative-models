import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

try:
  from misc import *
  from decoder_generator_base import DecoderGenerator
except ImportError:
  from .misc import *
  from .decoder_generator_base import DecoderGenerator

class Wasserstein_GAN(DecoderGenerator):

  def __init__(self, data_sampler, **kw):
    super().__init__(data_sampler, **kw)
    self._tf_call_kw           = retrieve_kw(kw, 'tf_call_kw',           {}                                                         )
    self._use_gradient_penalty = retrieve_kw(kw, 'use_gradient_penalty', True                                                       )
    self._grad_weight          = tf.constant( retrieve_kw(kw, 'grad_weight',          10.0                                          ) )
    self._lkeys |= {"lipschitz"}

  def latent_dim(self):
    return self._latent_dim

  @tf.function
  def latent_log_prob(self, latent):
    prior = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(self._latent_dim),
                                                     scale_diag=tf.ones(self._latent_dim))
    return prior.log_prob(latent)

  @tf.function
  def wasserstein_loss(self, y_true, y_pred):
    return tf.reduce_mean(y_true) - tf.reduce_mean(y_pred)

  @tf.function
  def sample_latent_data(self, nsamples):
    return tf.random.normal((nsamples, self._latent_dim))

  @tf.function
  def transform(self, latent, **call_kw):
    return self.generator( latent, **call_kw)

  @tf.function
  def generate(self, nsamples, **call_kw):
    return self.transform( self.sample_latent_data( nsamples ),**call_kw)


  @tf.function
  def _gradient_penalty(self, x, x_hat):
    epsilon = tf.random.uniform((x.shape[0], 1, 1), 0.0, 1.0)
    u_hat = epsilon * x + (1 - epsilon) * x_hat
    with tf.GradientTape() as penalty_tape:
      penalty_tape.watch(u_hat)
      func = self.critic(u_hat)
    grads = penalty_tape.gradient(func, u_hat)
    norm_grads = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
    regularizer = tf.math.square( tf.reduce_mean((norm_grads - 1) ) )
    return regularizer

  @tf.function
  def _get_critic_output( self, samples, fake_samples ):
    # calculate critic outputs
    real_output = self.critic(samples, **self._training_kw)
    fake_output = self.critic(fake_samples, **self._training_kw)
    return real_output, fake_output

  @tf.function
  def _get_critic_loss( self, samples, fake_samples, real_output, fake_output ):
    critic_lipschitz = tf.multiply(self._grad_weight
        , self._gradient_penalty(samples, fake_samples)
        ) if self._use_gradient_penalty else 0
    critic_loss = tf.add( self.wasserstein_loss(real_output, fake_output), critic_lipschitz )
    return critic_loss, critic_lipschitz

  def _get_gen_loss( self, fake_samples, fake_output ):
    gen_loss = tf.reduce_mean(fake_output)
    return gen_loss

  def _apply_critic_update( self, critic_tape, critic_loss ):
    critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
    if self._use_grad_clipping:
      critic_grads = [self._grad_clipping_fcn(g) for g in critic_grads]
    self._critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
    return

  def _apply_gen_update( self, gen_tape, gen_loss):
    gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
    if self._use_grad_clipping:
      gen_grads = [self._grad_clipping_fcn(g) for g in gen_grads]
    self._gen_opt.apply_gradients(zip(gen_grads, self.generator.trainable_variables))
    return

  @tf.function
  def _train_step(self, samples, fake_cond_samp = None, critic_only = False):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as critic_tape:
      fake_samples = self.generate( self.data_sampler._batch_size, **self._training_kw )
      real_output, fake_output = self._get_critic_output( samples, fake_samples )
      critic_loss, critic_lipschitz = self._get_critic_loss( samples, fake_samples
                                                           , real_output, fake_output )
      if not critic_only:
        gen_loss = self._get_gen_loss( fake_samples, fake_output )
    # gen_tape, critic_tape
    self._apply_critic_update( critic_tape, critic_loss )
    ret_dict = { 'critic' :         critic_loss
               , 'lipschitz' :      critic_lipschitz }
    if not critic_only:
      self._apply_gen_update( gen_tape, gen_loss )
      ret_dict['generator'] = gen_loss
    return ret_dict
