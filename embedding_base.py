import os
import numpy as np
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

class SelectFeatures(layers.Layer):
  def __init__(self, input_shape, start, n, **kw):
    self.start = start
    self.end = start + n
    layers.Layer.__init__(self, **kw)

  def call(self, inputs):
    return inputs[:,self.start:self.end]

class EmbeddingConfig(object):
  def __init__(self
        , input_info
        , embedding_layer_kwargs  = {}
        , embedding_size_fcn = None
        ):
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
        dim = int( input_info.n_variables // 2 )
    return dim

  @property
  def dim( self ):
    dim = self.embedding_size_fcn(self.input_info)
    if bool(dim) and dim <= 1:
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

  def use_default_marginal_statistics_bias(self, data, mask):
    acc = np.sum(data, axis=0)
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
    statistics = ( acc / div )
    bias = inverse_function( statistics )
    bias = np.where( np.isfinite( bias ), bias, np.zeros_like( bias ) )
    name = self.category_name if hasattr(self,"category_name") else "NumericalInputs"
    print("Assigning %s biases to %r...\n...in order to get marginal statistics %r" % (name, bias, statistics))
    self.bias = bias
    #self.output_layer_kwargs["bias_initializer"] = BiasInitializer(biases)

class ModelWithEmbeddings( BeyondNumericalDataModel ):

  def __init__(self
      , input_info_dict
      , embeddings_master_switch = True
      , output_head_hidden_layer_master_switch = True
      , output_head_bias_statistics_master_switch = True
      , **kw):
    BeyondNumericalDataModel.__init__(self, input_info_dict, **kw)
    self._embeddings_master_switch = embeddings_master_switch
    self._output_head_hidden_layer_master_switch = output_head_hidden_layer_master_switch 
    self._output_head_bias_statistics_master_switch = output_head_bias_statistics_master_switch

  def _create_initial_layers( self, train_data, train_mask, embedding_config_dict = NotSet, embedding_size_fcn = NotSet ):
    if not embedding_config_dict:
      embedding_config_dict = { k : EmbeddingConfig( input_info = v, embedding_size_fcn = embedding_size_fcn ) for k, v in self._input_info_dict.items() }
    self._embedding_config_dict = embedding_config_dict
    import unidecode
    model_inputs = []
    raw_inputs = []
    #
    sigmoid_inputs = []
    softmax_inputs = []
    numerical_inputs = []
    #
    softmax_mask_slice = []
    sigmoid_mask_slice = []
    numerical_mask_slice = []
    #
    flatten_input = layers.Input(shape=(self._n_features,))
    c_input = 0; c_mask_input = 0;
    for name, info, econf in zip(self._input_info_dict.keys(), self._input_info_dict.values(), self._embedding_config_dict.values()):
      # Select features:
      name = unidecode.unidecode(name).replace(' ','_') + "_Select"
      raw_in = SelectFeatures(input_shape=flatten_input.shape, start=c_input, n=info.n_variables, name = name)(flatten_input)
      # Add embedding:
      if econf and self._embeddings_master_switch:
        if isinstance(info, CategoricalInputInfoBase):
          #if info.already_as_one_hot:
            model_input = layers.Dense(econf.dim
                , input_dim = info.n_variables
                , **econf.embedding_layer_kwargs)(raw_in)
          #else:
          #  model_input = layers.Embedding( input_dim = info.n_variables
          #      , output_dim = econf.dim 
          #      )
        else:
          model_input = layers.Dense(econf.dim
              , input_dim = info.n_variables
              , **econf.embedding_layer_kwargs)(raw_in)
      else:
        model_input = raw_in
      # Register input category
      if isinstance(info, CategoricalInputInfoBase):
        if isinstance(info, BinaryInputInfo):
          sigmoid_inputs.append(raw_in)
          sigmoid_mask_slice.append(c_mask_input)
        elif isinstance(info, CategoricalGroupInputInfo):
          softmax_inputs.append(raw_in)
          softmax_mask_slice.append(c_mask_input)
        c_mask_input += 1
      else:
        numerical_inputs.append(raw_in)
        numerical_mask_slice += list(range(c_mask_input, c_mask_input + info.n_variables))
        c_mask_input += info.n_variables
      model_inputs.append(model_input)
      raw_inputs.append(raw_in)
      c_input += info.n_variables
    if len(model_inputs) > 1:
      model_input = layers.Concatenate( axis = -1)( model_inputs )
    else:
      model_input = model_input
    self._raw_inputs    = raw_inputs
    self._model_input   = model_input
    self._flatten_input = flatten_input
    # 
    self._sigmoid_input_layers  = sigmoid_inputs
    startpos = 0
    endpos = startpos+len(sigmoid_inputs)
    self._sigmoid_input_slice = slice(startpos,endpos)
    self._softmax_input_layers = softmax_inputs
    startpos = endpos
    endpos = startpos+len(softmax_inputs)
    self._softmax_input_slice = slice(startpos,endpos)
    self._numerical_input_layer = layers.Concatenate( axis = -1)( numerical_inputs ) if len(numerical_inputs) > 1 else numerical_inputs[0]
    startpos = endpos
    endpos = startpos+1
    self._numerical_input_slice = slice(startpos,endpos)
    self._input_end_pos = endpos
    #
    self._softmax_mask_select,  self._softmax_mask_shape    = self._create_mask_from_slice( softmax_mask_slice   )
    self._sigmoid_mask_select, self._sigmoid_mask_shape     = self._create_mask_from_slice( sigmoid_mask_slice  )
    self._numerical_mask_select, self._numerical_mask_shape = self._create_mask_from_slice( numerical_mask_slice )
    self._has_softmax   = tf.constant( len(softmax_mask_slice) > 0,   tf.bool )
    self._has_sigmoid   = tf.constant( len(sigmoid_mask_slice) > 0,  tf.bool )
    self._has_numerical = tf.constant( len(numerical_mask_slice) > 0, tf.bool )
    return flatten_input, model_input

  def _create_final_layers( self, final_codes, train_data, train_mask, output_head_config_dict = NotSet
      , hidden_layer_activation_type = NotSet
      , output_head_hidden_model_config_fcn = NotSet
      , use_batch_normalization = False
      , use_dropout = False  ):
    import unidecode
    # TODO output_head_config should bring inside the OutputHeadConfig the full processing chain for each head. I.e. remove batch normalization etc
    if not output_head_config_dict:
      output_head_config_dict = { k : OutputHeadConfig( 
        input_info = v, embedding_config = e, output_head_hidden_model_config_fcn = output_head_hidden_model_config_fcn )
        for (k, v), e in zip(self._input_info_dict.items(), self._embedding_config_dict.values() ) }
    self._output_head_config_dict = output_head_config_dict
    dense_final_code = False
    if not isinstance(final_codes, (tuple,list)):
      dense_final_code = True
      final_codes = [final_codes]*len(self._input_info_dict)
    assert len(final_codes) == len(self._input_info_dict)
    n_variables = 0
    outputs = []
    sigmoid_logits = []
    softmax_logits = []
    numerical_outputs = []
    for name, info, oConf, code in zip(self._input_info_dict.keys(), self._input_info_dict.values(), self._output_head_config_dict.values(), final_codes):
      name = unidecode.unidecode(name).replace(' ','_') + "_Head"
      if oConf.output_n_hidden and self._output_head_hidden_layer_master_switch:
        # FIXME Improve (just place the hidden model from oConf here)
        # The hidden layer
        output = layers.Dense(oConf.output_n_hidden
            , name = name
            , **oConf.output_hidden_layer_kwargs )(code)
        if use_batch_normalization:
          output = layers.BatchNormalization()(output)
        output = layers.Activation( hidden_layer_activation_type )(output)
        if use_dropout:
          output = layers.Dropout(rate=0.1)(output)
        # The output layer
        output = layers.Dense( info.n_variables,
            name = name + ("_Logits" if isinstance(info, CategoricalInputInfoBase) else "_Outputs")
            , **oConf.output_layer_kwargs
            )(output)
      else:
        # The output layer
        output = layers.Dense( info.n_variables,
            name = name + ("_Logits" if isinstance(info, CategoricalInputInfoBase) else "_Outputs")
            , **oConf.output_layer_kwargs
            )(code)
      # Fix marginal statistics using bias?
      if train_data is not None and oConf.use_marginal_statistics and self._output_head_hidden_layer_master_switch:
        oConf.use_default_marginal_statistics_bias( 
            train_data[:,n_variables:n_variables+info.n_variables],
            self._expand_mask( train_mask )[:,n_variables:n_variables+info.n_variables]
        )
        output = layers.Dense(info.n_variables
            , name = name + "_marginal_statistics_fixer"
            , weights = [tf.eye(info.n_variables), oConf.bias]
            , trainable = False )(output)
      # Output activation
      if oConf.output_activation:
        # Add the logits 
        if oConf.output_activation is tf.keras.activations.sigmoid:
          sigmoid_logits.append(output)
        elif oConf.output_activation is tf.keras.activations.softmax:
          softmax_logits.append(output)
        output = layers.Activation( oConf.output_activation
            , name = oConf.output_activation.__name__ + "_" + str(n_variables) + "_" + str(n_variables+info.n_variables)
            )(output)
      if not(isinstance(info, CategoricalInputInfoBase)):
        numerical_outputs.append(output)
      outputs.append(output)
      n_variables += info.n_variables
    flatten_output = layers.Concatenate( axis = -1)( outputs ) if len(outputs) > 1 else output
    self._flatten_output = flatten_output
    self._raw_outputs = outputs
    #
    self._sigmoid_logits_layers  = sigmoid_logits
    startpos = self._input_end_pos
    endpos = startpos+len(sigmoid_logits)
    self._sigmoid_logits_slice = slice(startpos,endpos)
    self._softmax_logits_layers   = softmax_logits
    startpos = endpos
    endpos = startpos+len(softmax_logits)
    self._softmax_logits_slice = slice(startpos,endpos)
    self._numerical_output_layer = layers.Concatenate( axis = -1)( numerical_outputs ) if len(numerical_outputs) > 1 else numerical_outputs[0]
    startpos = endpos
    endpos = startpos+1
    self._numerical_output_slice = slice(startpos,endpos)
    del(self._input_end_pos)
    return flatten_output

