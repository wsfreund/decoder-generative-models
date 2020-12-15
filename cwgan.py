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

class cWasserstein_GAN(cDecoderGenerator, Wasserstein_GAN):

  def __init__(self, data_sampler, **kw):
    super().__init__(data_sampler, **kw)

  def transform(self, gen_batch, **call_kw):
    return self.generator( gen_batch, **call_kw )

  def generate(self, n_cond = 1, n_fakes = 10, ds = 'val'):
    # Sample conditions
    sampler = self.data_sampler
    all_generated = []
    all_conditions = []
    all_samples = []
    for _ in range(n_cond):
      raw_sample = self.sample_parser_fcn( sampler.sample( n_samples = 1, ds = ds ) )
      raw_sample = self._ensure_batch_size_dim(self.generator, raw_sample)
      samples = [tf.repeat( x, [n_fakes], axis = 0 ) for x in raw_sample ]
      generator_input = self.sample_generator_input(sampled_input = samples)
      generated_samples = self.transform( generator_input )
      all_generated.append(generated_samples)
      all_samples.append(raw_sample)
    return ( tf.stack( all_generated )  if n_cond > 1 else all_generated[0]
           , tf.stack( all_samples )    if n_cond > 1 else all_samples[0] )

  def extract_critic_input(self, data):
    """Only required when applying lipschitz smoothening"""
    return data[1]

  def extract_critic_conditioning(self, data):
    """Only required when applying lipschitz smoothening"""
    return data[0]

  @tf.function
  def _compute_u_hat(self, x, x_hat):
    x_shape = tf.concat([x.shape[:1],tf.ones_like(x.shape[1:],dtype=tf.int32)],axis=0)
    epsilon = tf.random.uniform(x_shape, 0.0, 1.0)
    u_hat = epsilon * x + (1 - epsilon) * x_hat
    return u_hat

  @tf.function
  def _lipschitz_penalty(self, x, x_hat, lipschitz_conditioning):
    u_hat = self._compute_u_hat(x, x_hat)
    with tf.GradientTape() as penalty_tape:
      penalty_tape.watch(u_hat)
      if not isinstance(lipschitz_conditioning, list):
        lipschitz_conditioning = [lipschitz_conditioning]
      critic_input = lipschitz_conditioning + [u_hat]
      func = self.critic(critic_input)
    grads = penalty_tape.gradient(func, u_hat)
    norm_grads = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=tf.range(1,tf.size(tf.shape(x)))))
    lipschitz = tf.math.square( tf.reduce_mean((norm_grads - 1)))
    lipschitz = tf.multiply(self._grad_weight, lipschitz)
    return lipschitz

  @tf.function
  def _get_critic_loss( self, real_output, fake_output, lipschitz ):
    critic_loss = tf.add( self.wasserstein_loss(real_output, fake_output), lipschitz )
    return critic_loss

  @tf.function
  def _split_lipschitz( self, real_batch, fake_batch ):
    real_idxs = tf.random.shuffle( tf.range(start = 0, limit = tf.constant(self.data_sampler._batch_size)) )[:self.data_sampler._batch_size//2]
    fake_idxs = tf.random.shuffle( tf.range(start = 0, limit = tf.constant(self.data_sampler._batch_size)) )[:self.data_sampler._batch_size//2]
    lipschitz_conditioning = [tf.concat([tf.gather(x,real_idxs)
                                        ,tf.gather(y,fake_idxs)], axis = 0)
                              for x, y in zip(self.extract_critic_conditioning(real_batch)
                                             ,self.extract_critic_conditioning(fake_batch))]
    real_inputs = tf.concat([tf.gather(self.extract_critic_input(real_batch), real_idxs)
                            ,tf.gather(self.extract_critic_input(fake_batch), fake_idxs)], axis = 0)
    sampled_input = tf.gather( self.extract_generator_input_from_standard_batch_fcn(real_batch), real_idxs) 
    if not isinstance(sampled_input, list):
      sampled_input = [sampled_input]
    generator_inpput = sampled_input + [self.sample_latent_data( self.data_sampler._batch_size//2 ) ]
    fake_inputs = tf.concat([self.transform( generator_input, **self._training_kw )
                            ,tf.gather(self.extract_critic_input(fake_batch), fake_idxs)], axis = 0)
    return real_inputs, fake_inputs, lipschitz_conditioning

  @tf.function
  def _train_step(self, real_batch, gen_batch, critic_only = False):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as critic_tape:
      fake_batch = self.transform(gen_batch, **self._training_kw )
      real_output, fake_output = self._get_critic_output( real_batch, fake_batch )
      if self._use_lipschitz_penalty: 
        if not(self.use_same_real_fake_conditioning):
          real_inputs, fake_inputs, lipschitz_conditioning = self._split_lipschitz( real_batch, fake_batch )
        else:
          real_inputs            = self.extract_critic_input(real_batch)
          fake_inputs            = self.extract_critic_input(fake_batch)
          lipschitz_conditioning = self.extract_critic_conditioning(real_batch)
        lipschitz = self._lipschitz_penalty(real_inputs, fake_inputs, lipschitz_conditioning)
      else:
        lipschitz = tf.constant(0.)
      critic_loss = self._get_critic_loss( real_output, fake_output, lipschitz )
      if not critic_only:
        gen_loss = self._get_gen_loss( fake_batch, fake_output )
    # gen_tape, critic_tape
    self._apply_critic_update( critic_tape, critic_loss )
    ret_dict = { 'critic' :    critic_loss
               , 'lipschitz' : lipschitz }
    if not critic_only:
      self._apply_gen_update( gen_tape, gen_loss )
      ret_dict['generator'] = gen_loss
    return ret_dict
