import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .misc import *

class Wasserstein_GAN(object):

  def __init__(self, **kw):
    self._max_epochs           = retrieve_kw(kw, 'max_epochs',           2000                                                       )
    self._batch_size           = retrieve_kw(kw, 'batch_size',           128                                                        )
    self._n_features           = retrieve_kw(kw, 'n_features',           NotSet                                                     )
    self._latent_dim           = retrieve_kw(kw, 'latent_dim',           100                                                        )
    self._n_critic             = retrieve_kw(kw, 'n_critic',               0                                                        )
    self._result_file          = retrieve_kw(kw, 'result_file',          NotSet                                                     )
    self._soft_label           = retrieve_kw(kw, 'soft_label',           False                                                      )
    self._save_interval        = retrieve_kw(kw, 'save_interval',        1000                                                       )
    self._use_gradient_penalty = retrieve_kw(kw, 'use_gradient_penalty', True                                                       )
    self._grad_weight          = retrieve_kw(kw, 'grad_weight',          10.0                                                       )
    self._verbose              = retrieve_kw(kw, 'verbose',              False                                                      )
    self._gen_opt              = retrieve_kw(kw, 'gen_opt',              tf.optimizers.Adam(lr=2e-4, beta_1=0.5, decay=1e-4 )       )
    self._critic_opt           = retrieve_kw(kw, 'critic_opt',           tf.optimizers.Adam(lr=2e-4, beta_1=0.5, decay=1e-4 )       )
    self._seed                 = retrieve_kw(kw, 'seed',                 None                                                       )
    self._tf_call_kw           = retrieve_kw(kw, 'tf_call_kw',           {}                                                         )

    # Initialize discriminator and generator networks
    self.critic = self._build_critic()
    self.generator = self._build_generator()

  def logp(self, latent):
    prior = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(self._latent_dim),
                                                     scale_diag=tf.ones(self._latent_dim))
    return prior.log_prob(latent)

  def wasserstein_loss(self, y_true, y_pred):
    return tf.reduce_mean(y_true) - tf.reduce_mean(y_pred)

  def sample_latent_data(self, nsamples):
    return tf.random.normal((nsamples, self._latent_dim))

  def transform(self, latent):
    return self.generator( latent, **self._tf_call_kw)

  def generate(self, nsamples):
    return self.transform( self.sample_latent_data( nsamples ))

  def train(self, train_data):
    if self._n_features is NotSet:
      self._n_features = train_data.shape[1]
    if self._verbose: print('Number of features is %d.' % self._n_features )
    gpus = tf.config.experimental.list_physical_devices('GPU')
    n_gpus = len(gpus)
    if self._verbose: print('This machine has %i GPUs.' % n_gpus)

    train_dataset = tf.data.Dataset.from_tensor_slices( train_data ).batch( self._batch_size, drop_remainder = True )

    # checkpoint for the model
    checkpoint_maker = tf.train.Checkpoint(generator_optimizer=self._gen_opt,
        discriminator_optimizer=self._critic_opt,
        generator=self.generator,
        discriminator=self.critic
    ) if self._result_file else None

    # containers for losses
    losses = {'critic': [], 'generator': [], 'regularizer': []}
    critic_acc = []

    #reduce_lr = ReduceLROnPlateau(monitor='loss', patience=100, mode='auto',
    #                              factor=factor, cooldown=0, min_lr=1e-4, verbose=2)
    #model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callback_list,
    #          class_weight=class_weight, verbose=2, validation_data=(X_test, y_test))
    updates = 0; batches = 0;
    for epoch in range(self._max_epochs):
      for sample_batch in train_dataset:
        if self._n_critic and (updates % self._n_critic):
          # Update only critic
          critic_loss, reg_loss, gen_loss = self._train_critic(sample_batch) + (np.nan,)
        if not(self._n_critic) or not(updates % self._n_critic):
          # Update critic and generator
          critic_loss, gen_loss, reg_loss = self._train_step(sample_batch)
        losses['critic'].append(critic_loss)
        losses['generator'].append(gen_loss)
        losses['regularizer'].append(reg_loss)
        updates += 1
        # Save current model
        if checkpoint_maker and not(updates % self._save_interval):
          checkpoint_maker.save(file_prefix=self._result_file)
          pass
        # Print logging information
        if self._verbose and not(epoch % 10) or not(updates % 1000):
          perc = np.around(100*epoch/self._max_epochs, decimals=1)
          print('Epoch: %i. Updates %i. Training %1.1f%% complete. Critic_loss: %.3f. Gen_loss: %.3f. Regularizer: %.3f'
               % (epoch, updates, perc, critic_loss, gen_loss, reg_loss ))
    return self.generator, losses

  def _gradient_penalty(self, x, x_hat):
    epsilon = tf.random.uniform((x.shape[0], 1, 1), 0.0, 1.0)
    u_hat = epsilon * x + (1 - epsilon) * x_hat
    with tf.GradientTape() as penalty_tape:
      penalty_tape.watch(u_hat)
      func = self.critic(u_hat)
    grads = penalty_tape.gradient(func, u_hat)
    norm_grads = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
    regularizer = tf.reduce_mean((norm_grads - 1) ** 2)
    return regularizer

  def _build_critic(self):
    raise NotImplementedError("Overload Wasserstein_GAN class with a critic model")

  def _build_generator(self):
    raise NotImplementedError("Overload Wasserstein_GAN class with a generator model")

  def _get_critic_output( self, samples, fake_samples ):
    # calculate critic outputs
    real_output = self.critic(samples, **self._tf_call_kw)
    fake_output = self.critic(fake_samples, **self._tf_call_kw)
    return real_output, fake_output

  def _get_critic_loss( self, samples, fake_samples, real_output, fake_output ):
    grad_regularizer_loss = (self._grad_weight * self._gradient_penalty(samples, fake_samples)) if self._use_gradient_penalty else 0
    critic_loss = self.wasserstein_loss(real_output, fake_output) + grad_regularizer_loss
    return critic_loss, grad_regularizer_loss

  def _get_gen_loss( self, fake_samples, fake_output ):
    gen_loss = tf.reduce_mean(fake_output)
    return gen_loss

  def _apply_critic_update( self, critic_tape, critic_loss ):
    critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
    self._critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
    return

  def _apply_gen_update( self, gen_tape, gen_loss):
    gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
    self._gen_opt.apply_gradients(zip(gen_grads, self.generator.trainable_variables))
    return

  @tf.function
  def _train_critic(self, samples):
    with tf.GradientTape() as critic_tape:
      fake_samples = self.generate( self._batch_size )
      real_output, fake_output = self._get_critic_output( samples, fake_samples )
      critic_loss, grad_regularizer_loss = self._get_critic_loss( samples, fake_samples, real_output, fake_output)
    # critic_tape
    self._apply_critic_update( critic_tape, critic_loss )
    return critic_loss, grad_regularizer_loss


  @tf.function
  def _train_step(self, samples):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as critic_tape:
      fake_samples = self.generate( self._batch_size )
      real_output, fake_output = self._get_critic_output( samples, fake_samples )
      critic_loss, critic_regularizer = self._get_critic_loss( samples, fake_samples, real_output, fake_output)
      gen_loss = self._get_gen_loss( fake_samples, fake_output )
    # gen_tape, critic_tape
    self._apply_critic_update( critic_tape, critic_loss )
    self._apply_gen_update( gen_tape, gen_loss )
    return critic_loss, gen_loss, critic_regularizer

