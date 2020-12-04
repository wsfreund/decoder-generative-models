import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

try:
  from misc import *
  from train_base import TrainBase
  from embedding_base import ModelWithEmbeddings
except ImportError:
  from .misc import *
  from .train_base import TrainBase
  from .embedding_base import ModelWithEmbeddings

class AutoEncoder( ModelWithEmbeddings ):

  def __init__(self, input_info_dict, train_data = None, train_mask = None, **kw):
    super().__init__(input_info_dict = input_info_dict, **kw )
    # Retrieve optimizer
    self._ae_opt = retrieve_kw(kw, 'ae_opt', tf.optimizers.Adam() ) 
    # Define loss keys
    # TODO Add loss conditionally to availability
    self._lkeys     |= {"ae_total","ae_numerical","ae_categorical", "ae_sigmoid", "ae_softmax"}
    self._val_lkeys |= {"ae_total","ae_numerical","ae_categorical", "ae_sigmoid", "ae_softmax", "step"}
    # Define early stopping key
    self.early_stopping_key = 'ae_total'
    self._val_prefix = 'ae'
    # Define loss function for performance evaluation (not training)
    self._loss_fcn = self.compute_loss
    # Compute mask values 
    ## build models
    self._train_model, encoder_final_layer = self._build_train_autoencoder( train_data, train_mask )
    self.reconstructor                        = self._build_reconstructor() 
    self.encoder                              = self._build_encoder( encoder_final_layer )
    # TODO return a decoder
    #self.decoder                           = fix_model_layers( self._build_decoder( encoder_final_layer )              )
    self._model_dict = { "ae_train_model" : self._train_model
                       , "ae_reconstructor" : self.reconstructor
                       , "ae_encoder" : self.encoder
                       #, "decoder" : self.decoder 
                       }

  @tf.function
  def encode(self, x, **call_kw):
    return self.encoder( x, **call_kw )

  @tf.function
  def decode(self, code, **call_kw):
    return self.decoder( code, **call_kw )

  @tf.function
  def reconstruction(self, x, **call_kw):
    return self.decode( self.encode( x, **call_kw ), **call_kw )

  def compute_loss(self, x, mask):
    return self._total_loss(x, mask, self._train_model, self._compute_ae_loss, prefix = "ae")

  def _build_train_autoencoder(self, train_data, train_mask ):
    raise NotImplementedError("Overload this method with training model")

  def _build_reconstructor(self):
    model = tf.keras.Model(self._flatten_input, self._flatten_output, name = "reconstructor")
    return model

  def _build_encoder(self, encoder_final_layer ):
    model = tf.keras.Model(self._flatten_input, encoder_final_layer, name = "encoder")
    model.compile()
    return model

  def _build_decoder(self, encoder_final_layer ):
    return None
    # TODO not urgent, and possibly difficult
    #code = layers.Input(shape=(self._n_code,))
    #flatten_output = self._flatten_output
    #model = tf.keras.Model(code, flatten_output, name = "decoder")
    #return model

  @tf.function
  def _compute_ae_loss( self, x, train_outputs, mask ):
    # NOTE Is it better to make it flow on the device or on standard CPU?
    if self._has_softmax:
      softmax_inputs    = train_outputs[self._softmax_input_slice]
      softmax_logits    = train_outputs[self._softmax_logits_slice]
      ae_softmax        = self._compute_softmax_loss( softmax_inputs, softmax_logits, mask )
    else:
      ae_softmax      = tf.constant(0., dtype=tf.float32 )
    if self._has_sigmoid:
      sigmoid_inputs    = train_outputs[self._sigmoid_input_slice]
      sigmoid_logits    = train_outputs[self._sigmoid_logits_slice]
      ae_sigmoid        = self._compute_sigmoid_loss( sigmoid_inputs, sigmoid_logits, mask )
    else:
      ae_sigmoid     = tf.constant(0., dtype=tf.float32 )
    if self._has_numerical:
      numerical_inputs  = train_outputs[self._numerical_input_slice][0]
      numerical_outputs = train_outputs[self._numerical_output_slice][0]
      ae_numerical    = self._compute_numerical_loss( numerical_inputs, numerical_outputs, mask )
    else:
      ae_numerical    = tf.constant(0., dtype=tf.float32 )
    ae_categorical    = tf.add( ae_softmax, ae_sigmoid )
    ae_total = tf.add( ae_numerical, ae_categorical )
    return { 'ae_total' :          ae_total
           , 'ae_numerical' :      ae_numerical
           , 'ae_categorical' :    ae_categorical
           , 'ae_sigmoid' :        ae_sigmoid 
           , 'ae_softmax' :        ae_softmax
           }

  #@tf.function
  def _apply_ae_update( self, ae_tape, ae_loss):
    ae_variables = self._train_model.trainable_variables
    ae_grads = ae_tape.gradient(ae_loss, ae_variables)
    if tf.cast( self._grad_clipping, tf.bool ):
      ae_grads = [tf.clip_by_value( layer_grads, -self._grad_clipping, self._grad_clipping ) 
          for layer_grads in ae_grads if layer_grads is not None]
    self._ae_opt.apply_gradients(zip(ae_grads, ae_variables))
    return

  @tf.function
  def _train_step(self, x, mask ):
    with tf.GradientTape() as ae_tape:
      train_outputs = self._train_model( x, **self._training_kw )
      ae_loss_dict = self._compute_ae_loss( x, train_outputs, mask, )
    # ae_tape,
    self._apply_ae_update( ae_tape, ae_loss_dict['ae_total'] )
    return ae_loss_dict
