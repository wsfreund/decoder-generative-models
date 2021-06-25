import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp

try:
  from misc import *
  from beyond_numerical import *
except ImportError:
  from .misc import *
  from .beyond_numerical import *

class BiasInitializer(tf.keras.initializers.Initializer):
  def __init__(self, bias_init_values):
    self.bias_init_values = bias_init_values

  def __call__(self, shape, dtype=None):
    return self.bias_init_values

class SelectFeatureSlice(layers.Layer):
  def __init__(self, input_shape, start, n, **kw):
    self.start = start
    self.end = start + n
    layers.Layer.__init__(self, **kw)

  def call(self, inputs):
    return inputs[:,self.start:self.end]

  def get_config(self):
    return {'start': self.start, 'end': self.end}

class SelectFeatureIndices(layers.Layer):
  def __init__(self, input_shape, indices, **kw):
    self.indices = indices
    layers.Layer.__init__(self, **kw)

  def call(self, inputs):
    return tf.gather(inputs, self.indices, axis=-1)

  def get_config(self):
    return {'indices': self.indices,}

class EmbeddingConfig(object):
  def __init__(self
        , input_info
        , embedding_layer_kwargs  = {}
        , embedding_size_fcn = None):
    if not "kernel_initializer" in embedding_layer_kwargs:
      embedding_layer_kwargs["kernel_initializer"] = tf.keras.initializers.RandomUniform(0,1)
    self.embedding_layer_kwargs = embedding_layer_kwargs
    self.input_info = input_info
    if not embedding_size_fcn: 
      embedding_size_fcn = self._default_embedding_size_fcn
    self.embedding_size_fcn = embedding_size_fcn

  def _default_embedding_size_fcn( self, input_info ):
    dim = None
    if isinstance( input_info, CategoricalInputInfoBase ):
      if input_info.n_variables > 10:
        e = int( input_info.n_variables // 2 )
        if e <= 50:
          dim = e
        else:
          dim = min(50, int(np.round( np.sqrt( input_info.n_variables ) * np.log10( input_info.n_variables ) ) ) )
      else:
        #self.dim = int(np.round( np.sqrt( self.input_info.n_variables ) ) )
        if input_info.n_variables == 3:
          dim = 2
        elif input_info.n_variables == 2:
          dim = 1
        else:
          dim = int( input_info.n_variables // 2 )
    return dim

  @property
  def dim( self ):
    dim = self.embedding_size_fcn(self.input_info)
    if bool(dim) and dim < 1:
      dim = None
    return dim

  def __bool__(self):
    return bool(self.dim) and self.dim > 1

class OutputHeadConfig(object):
  def __init__(self
        , input_info
        , embedding_config = NotSet
        , output_head_hidden_model_config_fcn = NotSet
        , output_hidden_layer_kwargs = {}
        , output_activation = NotSet
        , output_layer_kwargs = {} 
        , use_marginal_statistics = True
        ):
    if not "kernel_initializer" in output_hidden_layer_kwargs:
      output_hidden_layer_kwargs["kernel_initializer"] = tf.keras.initializers.RandomUniform(0,1)
    self.output_hidden_layer_kwargs = output_hidden_layer_kwargs
    self.output_layer_kwargs = output_layer_kwargs
    self.output_activation = output_activation
    if self.output_activation is NotSet:
      if isinstance(input_info, CategoricalInputInfoBase):
        if isinstance(input_info, BinaryInputInfo):
          self.output_activation = tf.keras.activations.sigmoid
        else:
          self.output_activation = tf.keras.activations.softmax
      elif isinstance(input_info, NumericalInputInfo):
        self.output_activation = tf.keras.activations.linear
    self.use_marginal_statistics = use_marginal_statistics
    self.input_info = input_info
    self.embedding_config = embedding_config
    if not output_head_hidden_model_config_fcn:
      output_head_hidden_model_config_fcn = self._default_output_head_hidden_model_config_fcn
    self._output_head_hidden_model_config_fcn = output_head_hidden_model_config_fcn

  @property
  def output_n_hidden( self ):
    embedding = self.embedding_config
    if not embedding:
      embedding = EmbeddingConfig( self.input_info )
    output_n_hidden = self._output_head_hidden_model_config_fcn(self.input_info, embedding )
    if bool(output_n_hidden) and output_n_hidden <= 1:
      output_n_hidden = None
    return output_n_hidden

  def _default_output_head_hidden_model_config_fcn( self, input_info, embedding_config ):
    if embedding_config.dim:
      output_n_hidden = embedding_config.dim + input_info.n_variables 
    else:
      output_n_hidden = None
    return output_n_hidden

  def use_default_marginal_statistics_bias(self, data, mask = None):
    acc = np.sum(data, axis=0)
    if mask is not None:
      div = np.sum(mask, axis=0)
      div[div==0.] = 1.
    if self.output_activation is tf.keras.activations.softmax:
      inverse_function = tfp.math.softplus_inverse
    elif self.output_activation is tf.keras.activations.sigmoid:
      inverse_function = lambda x: np.log( x / (1 - x) )
    elif self.output_activation is tf.keras.activations.tanh:
      inverse_function = lambda x: 0.5*np.log( (1 + x) / (1 - x) )
    elif self.output_activation is tf.keras.activations.linear:
      inverse_function = lambda x: x
    if mask is not None:
      statistics = ( acc / div )
    bias = inverse_function( statistics )
    bias = np.where( np.isfinite( bias ), bias, np.zeros_like( bias ) )
    name = self.category_name if hasattr(self,"category_name") else "NumericalInputs"
    print("Assigning %s biases to %r...\n...in order to get marginal statistics %r" % (name, bias, statistics))
    self.bias = bias
    #self.output_layer_kwargs["bias_initializer"] = BiasInitializer(biases)

class ModelWithEmbeddings( BeyondNumericalDataModel ):

  def __init__(self, data_sampler, **kw):
    self._embeddings_master_switch                  = retrieve_kw( kw, 'embeddings_master_switch',                  True )
    self._output_head_hidden_layer_master_switch    = retrieve_kw( kw, 'output_head_hidden_layer_master_switch',    True )
    self._output_head_bias_statistics_master_switch = retrieve_kw( kw, 'output_head_bias_statistics_master_switch', True )
    super().__init__( data_sampler, **kw )

  def get_batch_size_from_data(self, data):
    return data['categorical']['data'].shape[0]

  def _create_binary_input_layers( self ):
    """
    Define binary_input, processed_binary_input and binary_config_dict. 
    Can be overloaded to create more complex input models.
    """
    import unidecode
    self.binary_input = layers.Input(shape=(self.data_sampler.n_binary_vars,), dtype=tf.int32, name = 'binary_inputs')
    self.binary_mask  = layers.Input(shape=(self.data_sampler.n_binary_vars,), dtype=tf.float32, name = 'binary_mask')
    self.processed_binary_input = []
    self._binary_config_dict = {}
    for idx, var_name in enumerate(self.data_sampler.binary_vars):
      name = unidecode.unidecode(var_name).replace(' ','_')
      raw_input_ = SelectFeatureIndices( input_shape=self.categorical_input.shape
                                       , indices = [idx]
                                       , name = name + '_Select')(self.binary_input)
      processed_input = tf.keras.layers.Lambda( lambda x: tf.cast(x, tf.float32), dtype = tf.float32 )( raw_input_ )
      self.processed_binary_input.append(processed_input)
      info = BinaryInputInfo( category_name    = name
                            , variable_names   = ['Not' + var_name,'Is' + var_name]
                            , variable_indices = [idx] )
      eConf = EmbeddingConfig( input_info = info )
      self._binary_config_dict[var_name] = eConf 
    return

  def _create_categorical_input_layers( self, embedding_config_dict = NotSet, embedding_size_fcn = NotSet ):
    """
    Define categorical_input, one_hot_layers, processed_categorical_input and categorical_config_dict
    Can be overloaded to create more complex input models
    """
    import unidecode
    self.categorical_input = layers.Input(shape=(self.data_sampler.n_categorical_vars,), dtype=tf.int32, name = 'categorical_inputs')
    self.categorical_mask = layers.Input(shape=(self.data_sampler.n_categorical_vars,), dtype=tf.float32, name = 'categorical_mask')
    self.one_hot_layers = []
    self.processed_categorical_input = []
    vocabulary = self.data_sampler.vocabulary
    one_hot_idx = 0
    self._categorical_config_dict = {} if embedding_config_dict in (NotSet,None) else embedding_config_dict
    for idx, var_name in enumerate(self.data_sampler.categorical_vars):
      name = unidecode.unidecode(var_name).replace(' ','_')
      # Select raw input
      raw_input_ = SelectFeatureIndices( input_shape=self.categorical_input.shape
                                       , indices = [idx]
                                       , name = name + "_Select")(self.categorical_input)
      # Check the categorical information:
      categories = vocabulary[var_name]
      n_cat = len(categories)
      lslice = slice(one_hot_idx,one_hot_idx+n_cat)
      one_hot_repr = OneHotEncodingLayerWithIntCategories(name = name + "_OneHot", depth=n_cat)(raw_input_)
      self.one_hot_layers.append(one_hot_repr)
      if var_name not in self._categorical_config_dict:
        info = CategoricalGroupInputInfo( category_name = name
                                        , variable_names = categories
                                        , variable_indices = lslice )
        eConf = EmbeddingConfig( input_info = info, embedding_size_fcn = embedding_size_fcn )
        self._categorical_config_dict[var_name] = eConf 
      one_hot_idx += n_cat
      # Add embedding:
      if self._embeddings_master_switch:
        processed_input = layers.Dense( eConf.dim, name = name + '_Embedding'
                                      , input_dim = (info.n_variables,)
                                      , **eConf.embedding_layer_kwargs)(one_hot_repr)
      else:
        processed_input = tf.keras.layers.Lambda( lambda x: tf.cast(x, tf.float32), dtype = tf.float32 )( one_hot_repr )
      self.processed_categorical_input.append(processed_input)
    return

  def _create_numerical_input_layers( self ):
    """
    Define numerical_input and processed_numerical_input. 
    Can be overloaded to create more complex input models.
    """
    self.numerical_input = layers.Input(shape=(self.data_sampler.n_numerical_vars,), name = 'numerical_inputs')
    self.numerical_mask  = layers.Input(shape=(self.data_sampler.n_numerical_vars,), name = 'numerical_mask')
    self.processed_numerical_input = self.numerical_input
    return

  def _create_initial_layers( self, embedding_config_dict = NotSet, embedding_size_fcn = NotSet ):
    # Process raw input
    self._create_categorical_input_layers( embedding_config_dict, embedding_size_fcn )
    self._create_binary_input_layers()
    self._create_numerical_input_layers()
    # Merge processed input
    processed_input_list = self.processed_binary_input + self.processed_categorical_input + [self.processed_numerical_input] 
    self._flatten_processed_input = layers.Concatenate( axis = -1 )( processed_input_list ) if len(processed_input_list) > 1 else processed_input_list[0]
    # Flag how to processed information
    self._has_sigmoid   = tf.constant( len(self.processed_binary_input) > 0, tf.bool )
    self._has_softmax   = tf.constant( len(self.processed_categorical_input) > 0,  tf.bool )
    self._has_numerical = tf.constant( self.data_sampler.n_numerical_vars > 0, tf.bool )
    # Create raw model input
    self.input = {}
    if self._has_sigmoid:   self.input['binary']      = {'data' : self.binary_input,      'mask' : self.binary_mask      }
    if self._has_softmax:   self.input['categorical'] = {'data' : self.categorical_input, 'mask' : self.categorical_mask }
    if self._has_numerical: self.input['numerical']   = {'data' : self.numerical_input,   'mask' : self.numerical_mask   }
    return self.input, self._flatten_processed_input

  def _retrieve_output_head_config(self, var_name, eConf, output_head_config_dict, output_head_hidden_model_config_fcn):
    if var_name not in output_head_config_dict:
      oConf = OutputHeadConfig( input_info = eConf.input_info
                              , embedding_config = eConf
                              , output_head_hidden_model_config_fcn = output_head_hidden_model_config_fcn )
    else:
      oConf = output_head_config_dict[var_name]
    return oConf

  def _output_head( self
                  , previous_final_output_layer
                  , use_batch_normalization
                  , hidden_layer_activation_type
                  , use_dropout
                  , oConf ):
    if oConf.output_n_hidden and self._output_head_hidden_layer_master_switch:
      output = layers.Dense( oConf.output_n_hidden, name = oConf.input_info.category_name + '_Hidden'
                           , **oConf.output_hidden_layer_kwargs )(previous_final_output_layer)
      if use_batch_normalization: output = layers.BatchNormalization(name = oConf.input_info.category_name + '_BN')(output)
      if hidden_layer_activation_type: output = layers.Activation( hidden_layer_activation_type, name = oConf.input_info.category_name + '_HiddenActivation' )(output)
      if use_dropout: output = layers.Dropout(rate=0.1)(output)
    else:
      output = previous_final_output_layer
    return output

  def _marginal_statistics_fixer( self, output, oConf ):
    if oConf.use_marginal_statistics and self._output_head_hidden_layer_master_switch:
      oConf.use_default_marginal_statistics_bias( 
          self.data_sampler.raw_train_data['categorical'].iloc[:,info.indices],
          #self._expand_mask( self.data_sampler.train_mask_df )[:,n_variables:n_variables+info.n_variables]
      )
      output = layers.Dense(info.n_variables, name = name + "_marginal_statistics_fixer"
          , weights = [tf.eye(info.n_variables), oConf.bias], trainable = False )(output)
    return output

  def _create_binary_output_layers( self
      , previous_final_output_layer
      , output_head_config_dict = NotSet
      , hidden_layer_activation_type = NotSet
      , output_head_hidden_model_config_fcn = NotSet
      , use_batch_normalization = False
      , use_dropout = False
      ):
    """
    Can be overloaded to create more complex output models
    """
    self.binary_logits = []
    self.binary_activation = []
    for idx, var_name in enumerate(self.data_sampler.binary_vars):
      eConf = self._binary_config_dict[var_name]
      oConf = self._retrieve_output_head_config( var_name, eConf, output_head_config_dict, output_head_hidden_model_config_fcn )
      output = self._output_head( previous_final_output_layer
                  , use_batch_normalization
                  , hidden_layer_activation_type
                  , use_dropout
                  , oConf )
      binary_logits_layer = layers.Dense( oConf.input_info.n_variables
                                        , name = oConf.input_info.category_name + "_Logits"
                                        , **oConf.output_layer_kwargs )
      binary_logits = binary_logits_layer(output)
      self.binary_logits.append(binary_logits)
      #output = self._marginal_statistics_fixer( output, oConf )
      activation_output = layers.Activation( oConf.output_activation
          , name = var_name + '_' + oConf.output_activation.__name__
      )(binary_logits)
      self.binary_activation.append(activation_output)
    return

  def _create_categorical_output_layers( self
      , previous_final_output_layer
      , output_head_config_dict = NotSet
      , hidden_layer_activation_type = NotSet
      , output_head_hidden_model_config_fcn = NotSet
      , use_batch_normalization = False
      , use_dropout = False
      ):
    """
    Can be overloaded to create more complex output models
    """
    self.categorical_logits = []
    self.categorical_activation = []
    for idx, var_name in enumerate(self.data_sampler.categorical_vars):
      eConf = self._categorical_config_dict[var_name]
      oConf = self._retrieve_output_head_config( var_name, eConf, output_head_config_dict, output_head_hidden_model_config_fcn )
      output = self._output_head( previous_final_output_layer
                  , use_batch_normalization
                  , hidden_layer_activation_type
                  , use_dropout
                  , oConf )
      categorical_logits_layer = layers.Dense( oConf.input_info.n_variables
                                             , name = oConf.input_info.category_name + "_Logits"
                                             , **oConf.output_layer_kwargs )
      categorical_logits = categorical_logits_layer(output)
      self.categorical_logits.append(categorical_logits)
      #output = self._marginal_statistics_fixer( output, oConf )
      activation_output = layers.Activation( oConf.output_activation
          , name = var_name + '_' + oConf.output_activation.__name__
      )(categorical_logits)
      self.categorical_activation.append(activation_output)
    return

  def _create_numerical_output_layer( self, previous_final_output_layer ):
    self.numerical_output = layers.Dense( self.numerical_input.shape[-1], name = 'NumericalOutputs')(previous_final_output_layer)
    return

  def _create_final_layers( self
      , previous_final_output_layer
      , output_head_config_dict = NotSet
      , hidden_layer_activation_type = NotSet
      , output_head_hidden_model_config_fcn = NotSet
      , use_batch_normalization = False
      , use_dropout = False  ):
    if output_head_config_dict in (NotSet,None):
      output_head_config_dict = {}
    kwargs = dict( output_head_config_dict = output_head_config_dict
                 , hidden_layer_activation_type = hidden_layer_activation_type 
                 , output_head_hidden_model_config_fcn = output_head_hidden_model_config_fcn 
                 , use_batch_normalization = use_batch_normalization
                 , use_dropout = use_dropout )
    self.output = {}
    if self._has_sigmoid:
      self._create_binary_output_layers( previous_final_output_layer, **kwargs )
      self.output['binary'] = self._concatenate_list_of_layers( self.binary_activation )
    if self._has_softmax:
      self._create_categorical_output_layers( previous_final_output_layer, **kwargs )
      self.output['categorical'] = self._concatenate_list_of_layers( self.categorical_activation )
    if self._has_numerical:
      self._create_numerical_output_layer( previous_final_output_layer )
      self.output['numerical'] = self.numerical_output 
    self._create_training_model()
    return self.output

  def _create_training_model(self):
    training_output_dict = { 'sigmoid_targets':   self.processed_binary_input
                           , 'softmax_targets':   self.one_hot_layers
                           , 'numerical_targets': self.processed_numerical_input
                           , 'sigmoid_logits':    self.binary_logits
                           , 'softmax_logits':    self.categorical_logits
                           , 'numerical_outputs': self.numerical_output }
    self._training_model = tf.keras.Model(self.input, training_output_dict, name = "training_model")
    return

  def _concatenate_list_of_layers( self, list_of_layers ):
    return layers.Concatenate( axis = -1 )( list_of_layers ) if len(list_of_layers) > 1 else list_of_layers[0]

class OneHotEncodingLayerWithIntCategories(layers.experimental.preprocessing.PreprocessingLayer):
  def __init__(self, depth, **kw):
    super().__init__(**kw)
    self.depth = depth

  def call(self,inputs):
    encoded = tf.one_hot(inputs, self.depth)
    return layers.Reshape((self.depth,))(encoded)

  def get_config(self):
    return {'depth': self.depth,}

class OneHotEncodingLayerWithStringCategories(layers.experimental.preprocessing.PreprocessingLayer):
  """
  Adapted from https://towardsdatascience.com/building-a-one-hot-encoding-layer-with-tensorflow-f907d686bf39
  """

  def __init__(self, vocabulary=None):
    super().__init__()
    self.vectorization = layers.experimental.preprocessing.TextVectorization(output_sequence_length=1)  
    self.adapt(vocabulary)

  def adapt(self, data):
    self.vectorization.adapt(data)
    vocab = self.vectorization.get_vocabulary()
    self.depth = len(vocab)
    indices = [i[0] for i in self.vectorization([[v] for v in vocab]).numpy()]
    self.minimum = min(indices)

  def call(self,inputs):
    vectorized = self.vectorization.call(inputs)
    subtracted = tf.subtract(vectorized, tf.constant([self.minimum], dtype=tf.int64))
    encoded = tf.one_hot(subtracted, self.depth)
    return layers.Reshape((self.depth,))(encoded)

  def get_config(self):
    return {'vocabulary': self.vectorization.get_vocabulary(), 'depth': self.depth, 'minimum': self.minimum}

