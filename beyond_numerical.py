import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

try:
  from misc import *
  from mask_base import MaskModel
  from train_base import TrainBase
except ImportError:
  from .misc import *
  from .mask_base import MaskModel
  from .train_base import TrainBase

class InputInfo(object):
  def __init__( self, variable_names
              , variable_slice ):
    self.variable_names = variable_names
    self.variable_slice = variable_slice

  @property
  def n_variables( self ):
    return len(self.variable_names)

  def __repr__( self ):
    return ( self.__class__.__name__ + "(" + str(self.variable_slice) + "|" + str(self.n_variables) + ")" )

class NumericalInputInfo(InputInfo):
  def __init__( self
              , variable_names
              , variable_slice ):
    InputInfo.__init__(self
        , variable_names             = variable_names
        , variable_slice             = variable_slice )

  @property
  def categorical_mask( self ):
    return [0]*self.n_variables

  @property
  def numerical_mask( self ):
    return [1]*self.n_variables

class CategoricalInputInfoBase(InputInfo):
  def __init__( self
      , category_name
      , variable_slice
      , variable_names ):
    self.category_name = category_name
    InputInfo.__init__(self
        , variable_names             = variable_names
        , variable_slice             = variable_slice )

  @property
  def categorical_mask( self ):
    return [1]*self.n_variables

  @property
  def numerical_mask( self ):
    return [0]*self.n_variables

  def __repr__( self ):
    return ( self.__class__.__name__ + "(" +  str(self.variable_slice) + "|var:" + self.category_name + "|"
        + str(self.n_variables) + ")" )


class BinaryInputInfo(CategoricalInputInfoBase):
  def __init__( self
      , category_name
      , variable_slice
      , variable_names ):
    self.category_name = category_name
    InputInfo.__init__(self
        , variable_names             = variable_names
        , variable_slice             = variable_slice )
    assert len(self.variable_names) == 2

  @property
  def n_variables( self ):
    return 1

class CategoricalGroupInputInfo(CategoricalInputInfoBase):
  def __init__( self
      , category_name
      , variable_slice
      , variable_names ):
    self.category_name = category_name
    InputInfo.__init__(self
        , variable_names             = variable_names
        , variable_slice             = variable_slice )
    assert len(self.variable_names) > 1

def hot_encode(df, categorical, keep_numerical = True):
  from scipy.sparse import coo_matrix, dok_matrix
  import pandas as pd
  input_info_dict = {}
  df_one_hot = pd.DataFrame()
  nan_arrays = {} 
  for i, c in enumerate(categorical):
    # Retrieve unique values:
    cat_val = list(set(df[c].dropna()))
    # Retrieve hot encodes for column
    df_tmp = pd.get_dummies( df[c]
        , dummy_na = df[c].isnull().values.any()
        , sparse=False, dtype=np.float32 )
        #, sparse=True, dtype=np.uint8 )
    if len(cat_val) == 1:
      print("Ignoring category %s for having only a single value" % c)
      continue
    if np.nan in df_tmp.columns:
      #ar = pd.Series(df_tmp[np.nan] == 0, dtype=pd.SparseDtype(np.float32,1))
      ##ar = pd.Series(df_tmp[np.nan] == 0, dtype=pd.SparseDtype(np.uint8,1))
      #if ar.sparse.density > .25:
      #  ar = pd.Series(df_tmp[np.nan] == 0, dtype=pd.SparseDtype(np.float32,0))
      ##  ar = pd.Series(df_tmp[np.nan] == 0, dtype=pd.SparseDtype(np.uint8,0))
      #  if ar.sparse.density > .25:
      ##    ar = pd.Series(df_tmp[np.nan] == 0, dtype=np.int8)
      ar = pd.Series(df_tmp[np.nan] == 0, dtype=np.float32)
      nan_arrays[c] = ar
      df_tmp = df_tmp.drop(columns=np.nan)
    else:
      #nan_arrays[c] = pd.Series(np.ones(shape=(df_tmp.shape[0],)), dtype=pd.SparseDtype(np.uint8,1))
      nan_arrays[c] = pd.Series(np.ones(shape=(df_tmp.shape[0],)), dtype=np.float32)
    # Retrieve formatted categories:
    colnms = ['%s_%s' % (c, el) for el in (cat_val + (['NaN'] if np.nan in df_tmp.columns else []))]
    # Expand one hot columns:
    if len(cat_val) > 2:
      #df_one_hot[colnms] = df_tmp
      df_one_hot[colnms] = df_tmp.astype(np.float32)
      info = CategoricalGroupInputInfo( category_name = c
        , variable_names = cat_val
        , variable_slice = slice(i,i+1)
      )
    else:
      #df_one_hot[colnms[1]] = df_tmp.iloc[:,1]
      df_one_hot[colnms[1]] = df_tmp.iloc[:,1].astype(np.float32)
      info = BinaryInputInfo( category_name = c
        , variable_names = cat_val
        , variable_slice = slice(i,i+1)
      )
    input_info_dict[c] = info
  if keep_numerical:
    numerical = list(filter(lambda c: c not in categorical, df.columns))
    # TODO check if numerical values are not None
    df_one_hot[numerical] = df[numerical]
    for n in numerical:
      query = df[n].notna() == 1
      ar = pd.Series(query, dtype=np.float32)
      ##ar = pd.Series(query, dtype=pd.SparseDtype(np.uint8,1))
      #ar = pd.Series(query, dtype=pd.SparseDtype(np.float32,1))
      #if ar.sparse.density > .25:
      ##  ar = pd.Series(query, dtype=pd.SparseDtype(np.uint8,0))
      #  ar = pd.Series(query, dtype=pd.SparseDtype(np.float32,0))
      #  if ar.sparse.density > .25:
      #    ar = pd.Series(query, dtype=np.float32)
      ##ar = pd.Series(query, dtype=np.int8)
      #  else:
      #    # Most data is nan, set it as sparse array
      #    df_one_hot[numerical] = df_one_hot[numerical].astype("Sparse")
      nan_arrays[n] = ar
    info = NumericalInputInfo( variable_names = numerical
      , variable_slice             = slice(i+1,i+1+len(numerical))
    )
    input_info_dict["NumericalVariables"] = info
  ##mask = pd.DataFrame( nan_arrays )
  mask = pd.DataFrame( nan_arrays ).astype(np.float32)
  df_one_hot = df_one_hot.fillna(0.)
  return df_one_hot, mask, input_info_dict

class BeyondNumericalDataModel(TrainBase):
  """
  Note: This class only works if batch dim is at tensor dimension 0
  """

  def __init__(self, input_info_dict, **kw):
    super().__init__(**kw)
    import itertools
    self._input_info_dict = input_info_dict
    # Compute feature type masks
    self._categorical_mask = tf.constant( 
      list( itertools.chain( *map(lambda i: i.categorical_mask, self._input_info_dict.values() ) ) )
    , dtype=tf.float32 )
    self._numerical_mask = tf.constant( 
      list( itertools.chain( *map(lambda i: i.numerical_mask, self._input_info_dict.values() ) ) )
    , dtype=tf.float32 )
    self._n_mask_inputs = tf.constant( 
          # Categorical information we only have a single mask for each one of them 
          sum([isinstance(i,CategoricalInputInfoBase) for i in self._input_info_dict.values()])
          # Numerical mask has an input for each feature
        + tf.cast(sum(self._numerical_mask), tf.int32)
    , dtype = tf.int32 )
    self._n_features = tf.constant( self._categorical_mask.shape[0], dtype = tf.int32 )
    self._expand_mask_matrix = self._retrieve_mask_mat()

  @tf.function
  def _compute_softmax_mask( self, mask ):
    n_mask_examples = tf.shape(mask)[0]
    return tf.reshape( mask[tf.tile(self._softmax_mask_select, [n_mask_examples,1])], [n_mask_examples] + self._softmax_mask_shape )

  @tf.function
  def _compute_sigmoid_mask( self, mask ):
    n_mask_examples = tf.shape(mask)[0]
    return tf.reshape( mask[tf.tile(self._sigmoid_mask_select, [n_mask_examples,1])], [n_mask_examples] + self._sigmoid_mask_shape )

  @tf.function
  def _compute_numerical_mask( self, mask ):
    n_mask_examples = tf.shape(mask)[0]
    return tf.reshape( mask[tf.tile(self._numerical_mask_select, [n_mask_examples,1])], [n_mask_examples] + self._numerical_mask_shape )

  @tf.function
  def _compute_numerical_loss( self, x, x_reco, mask):
    x, mask = self._retrieve_data_and_mask( x )
    if mask is not None:
      mask = self._compute_numerical_mask( mask )
    reco_numerical = self._reduce_mean_mask( 
      tf.square( 
        tf.subtract( x, x_reco )
      )
    , mask ) if mask is None or tf.reduce_any(tf.cast(mask, tf.bool)) else tf.constant(0., dtype=tf.float32)
    return reco_numerical

  @tf.function
  def _compute_sigmoid_loss( self, labels, logits, mask):
    x, mask = self._retrieve_data_and_mask( x )
    if mask is not None:
      mask = self._compute_sigmoid_mask( mask )
      if tf.math.logical_not(tf.reduce_any(tf.cast(mask, tf.bool))):
        return tf.constant(0., dtype=tf.float32)
    loss = []; count = 0;
    for i, (label, logit) in enumerate(zip(labels, logits)):
      per_example_loss = tf.expand_dims( tf.squeeze( tf.nn.sigmoid_cross_entropy_with_logits(
            labels = label,
            logits = logit ) ), axis = 1 )
      m = tf.expand_dims( mask[:,i], axis = 1 ) if mask is not None else None
      category_loss = tf.squeeze( 
          self._reduce_mean_mask_per_example( 
            per_example_loss 
          , m ) 
      )
      loss.append( category_loss )
      count += tf.cast( tf.reduce_any(tf.cast(m, tf.bool)), tf.float32 ) if mask is not None else tf.shape( x )[0]
    tot = tf.math.divide_no_nan( tf.reduce_sum( loss ), count )
    return tot

  @tf.function
  def _compute_softmax_loss( self, labels, logits, mask):
    x, mask = self._retrieve_data_and_mask( x )
    if mask is not None:
      mask = self._compute_softmax_mask( mask )
      if tf.math.logical_not(tf.reduce_any(tf.cast(mask, tf.bool))):
        return tf.constant(0., dtype=tf.float32)
    loss = []; count = 0;
    for i, (label, logit) in enumerate(zip(labels, logits)):
      per_example_loss = tf.expand_dims( tf.nn.softmax_cross_entropy_with_logits(
            labels = label,
            logits = logit ), axis = 1 )
      m = tf.expand_dims( mask[:,i], axis = 1 ) if mask is not None else None
      category_loss = tf.squeeze( 
          self._reduce_mean_mask_per_example( 
            per_example_loss 
          , m ) 
      )
      loss.append( category_loss )
      count += tf.cast( tf.reduce_any(tf.cast(m, tf.bool)), tf.float32 ) if mask is not None else tf.shape( x )[0]
    tot = tf.math.divide_no_nan( tf.reduce_sum( loss ), count )
    return tot

  def _parse_surrogate_loss(self, train_loss, prefix = ''):
    # TODO Make a set of prefix
    if prefix and not(prefix.endswith('_')): prefix += '_'
    train_loss = TrainBase._parse_surrogate_loss(self, train_loss)
    if (prefix + 'numerical') in train_loss:
      train_loss[prefix + 'numerical'] = np.sqrt(train_loss[prefix + 'numerical'])
      train_loss[prefix + 'total'] = train_loss[prefix + 'numerical'] + train_loss[prefix + 'categorical']
    return train_loss

  def _retrieve_mask_mat( self ):
    mat = np.zeros((self._n_mask_inputs,self._n_features), dtype=np.float32)
    l = 0; c = 0
    for info in self._input_info_dict.values():
      n=info.n_variables
      if isinstance(info, CategoricalInputInfoBase):
        mat[l,c:(c+n)] = 1.
        l+=1
      else:
        for l2 in range(n):
          mat[l+l2,c+l2] = 1.
        l+=l2
      c+=n
    return tf.constant( mat, dtype=tf.float32 )

  @tf.function
  def _expand_mask( self, mask ):
    return tf.linalg.matmul( mask, self._expand_mask_matrix )

