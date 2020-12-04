import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

try:
  from misc import *
  from conditional_decoder_generator import cDecoderGenerator
  from wgan import Wasserstein_GAN
except ImportError:
  from .misc import *
  from .conditional_decoder_generator import cDecoderGenerator
  from .wgan import Wasserstein_GAN

class cWasserstein_GAN(Wasserstein_GAN, cDecoderGenerator):

  def __init__(self, data_sampler, **kw):
    super().__init__(data_sampler, **kw)

  def transform(self, latent, fake_cond_samp = None, **call_kw):
    gen_input = latent if fake_cond_samp is None else (fake_cond_samp, latent)
    return self.generator( gen_input, **call_kw )

  def generate(self, n_cond = 1, n_fakes = 10, ds = 'val'):
    # Sample conditions
    sampler = self.data_sampler
    all_generated = []
    all_conditions = []
    all_samples = []
    for _ in range(n_cond):
      raw_sample = sampler.sample( ds )
      sample = self._sample_parser_fcn( raw_sample )
      condition = self._extract_conditioning_fcn( sample )
      generated_samples = self.transform( self.sample_latent_data( n_fakes )
                                        , tf.repeat( condition, [n_fakes], axis = 0 ) )
      all_generated.append(generated_samples)
      all_samples.append(sample)
      all_conditions.append(condition)
    return ( tf.stack( all_generated ) if n_cond > 1 else all_generated[0]
           , tf.stack( all_conditions ) if n_cond > 1 else all_conditions[0]
           , tf.stack( all_samples ) if n_cond > 1 else all_samples[0] )

  @tf.function
  def _gradient_penalty(self, x, x_hat, lipschitz_conditioning):
    epsilon = tf.random.uniform((x.shape[0], 1, 1), 0.0, 1.0)
    u_hat = epsilon * x + (1 - epsilon) * x_hat
    with tf.GradientTape() as penalty_tape:
      penalty_tape.watch(u_hat)
      critic_input = (lipschitz_conditioning, u_hat)
      func = self.critic(critic_input)
    grads = penalty_tape.gradient(func, u_hat)
    norm_grads = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
    regularizer = tf.math.square( tf.reduce_mean((norm_grads - 1) ) )
    return regularizer

  @tf.function
  def _get_critic_loss( self, samples, fake_samples, real_output, fake_output, lipschitz_conditioning ):
    critic_lipschitz = tf.multiply(self._grad_weight
        , self._gradient_penalty(samples, fake_samples, lipschitz_conditioning)
        ) if self._use_gradient_penalty else 0
    critic_loss = tf.add( self.wasserstein_loss(real_output, fake_output), critic_lipschitz )
    return critic_loss, critic_lipschitz

  @tf.function
  def _split_lipschitz( self, samples, additional_samples, fake_samples ):
    real_idxs = tf.random.shuffle( tf.range(start = 0, limit = tf.constant(self.data_sampler._batch_size)) )[:self.data_sampler._batch_size//2]
    fake_idxs = tf.random.shuffle( tf.range(start = 0, limit = tf.constant(self.data_sampler._batch_size)) )[:self.data_sampler._batch_size//2]
    lipschitz_conditioning = tf.concat([tf.gather(samples[0],real_idxs), tf.gather(additional_samples[0],fake_idxs)], axis = 0)
    real_inputs = tf.concat([tf.gather(samples[1], real_idxs),tf.gather(additional_samples[1], fake_idxs)], axis = 0)
    fake_inputs = tf.concat([self.transform( self.sample_latent_data( self.data_sampler._batch_size//2 )
                                           , tf.gather(samples[0], real_idxs))
                            , tf.gather(fake_samples, fake_idxs)], axis = 0)
    return real_inputs, fake_inputs, lipschitz_conditioning

  @tf.function
  def _train_step(self, samples, additional_samples, critic_only = False):
    fake_cond_samples = additional_samples if self._use_same_real_fake_conditioning else additional_samples[0]
    with tf.GradientTape() as gen_tape, tf.GradientTape() as critic_tape:
      fake_samples = self.transform( self.sample_latent_data( self.data_sampler._batch_size ), fake_cond_samples, **self._training_kw )
      full_fake_samples = (fake_cond_samples, fake_samples)
      real_output, fake_output = self._get_critic_output( samples, full_fake_samples )
      if not(self._use_same_real_fake_conditioning):
        real_inputs, fake_inputs, lipschitz_conditioning = self._split_lipschitz( samples, additional_samples, fake_samples )
      else:
        real_inputs, fake_inputs, lipschitz_conditioning = samples[1], fake_samples, samples[0]
      critic_loss, critic_lipschitz = self._get_critic_loss( real_inputs, fake_inputs
                                                           , real_output, fake_output
                                                           , lipschitz_conditioning = lipschitz_conditioning )
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
