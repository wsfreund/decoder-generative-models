from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
import datetime
import copy
import itertools

from ..misc import *
from .sampler_base import SamplerBase, SpecificFlowSamplingOpts

class _CacheStorage(CleareableCache):
  cached_functions = []

class DataFrameSampler(SamplerBase):

  def __init__(self, manager, **kw ):
    """
    TODO Help
    :param manager: a data manager instance 
    """
    self._label = retrieve_kw(kw, 'label', None )
    #self._keep_vars_unused_to_fit = retrieve_kw(kw, 'keep_vars_unused_to_fit', False )
    #self._vars_to_drop_on_fit = retrieve_kw(kw, 'keep_vars_unused_to_fit', False )
    self._fill_metadata( manager.df )
    super().__init__( raw_data = manager.df, **kw)
    self._pp = manager.pre_proc if hasattr(manager,'pre_proc') else None
    del kw
    return

  def clear_cache(self):
    # FIXME
    _CacheStorage.clear_cached_functions()
    SamplerBase.clear_cache(self)

  def plot(self, model = None,  ds = "val", n_examples = 3):
    return

  def _retrieve_vocabulary( self, df ):
    try:
      self.vocabulary = df.attrs['vocabulary']
    except KeyError:
      print('WARNING: No vocabulary available. Continuing using codes as vocabulary.')
      temp_categorical_vars = [v for v, t in zip(df.dtypes.index, df.dtypes) if pd.api.types.is_integer_dtype(t)]
      self.vocabulary = {v : {vu : str(vu) for vu in df[v].unique()} for v in temp_categorical_vars}

  def _fill_metadata( self, df ):
    # TODO Also compute number of categories based on the number of unique integers available in each variable
    self._retrieve_vocabulary( df )
    temp_categorical_vars = [v for v, t in zip(df.dtypes.index, df.dtypes) if pd.api.types.is_integer_dtype(t)]
    self.n_categories     = {v : len(df[v].unique()) for v in temp_categorical_vars}
    self.categorical_vars = [v for v, n in self.n_categories.items() if n > 2]
    self.binary_vars      = [v for v, n in self.n_categories.items() if n == 2]
    self.numerical_vars   = [v for v, t in zip(df.dtypes.index, df.dtypes) if pd.api.types.is_float_dtype(t)]
    self.n_categorical_vars = len(self.categorical_vars)
    self.n_binary_vars      = len(self.binary_vars)
    self.n_numerical_vars   = len(self.numerical_vars)
    assert len(set(self.categorical_vars)) == self.n_categorical_vars
    assert len(set(self.binary_vars)) == self.n_binary_vars
    assert len(set(self.numerical_vars)) == self.n_numerical_vars

  def _to_numpy( data ):
    if isinstance(df, dict):
      data = {k : d.to_numpy() for k, d in df.items()}
    else:
      data = df.to_numpy()

  def _to_numpy(self, data):
    if isinstance(data, dict):
      return {k : self._to_numpy(v) for k, v in data.items()}
    elif isinstance(data,(tuple,list)):
      return [self._to_numpy(v) for v in data]
    elif data is None:
      return data
    else:
      return data.to_numpy()


  def _make_dataset( self, df, opts, cache_filepath ):
    #start = datetime.datetime.now()
    #print("Building new dataset...")
    # This will not work if attempting to forecast the past:
    try:
      if self._pp:
        sklearn.utils.validation.check_is_fitted( self._pp )
    except sklearn.exceptions.NotFittedError:
      self._pp.fit(self.raw_train_data.to_numpy())
    data = self._to_numpy( df ) 
    opts.set_unset_to_default( self, data )
    # NOTE This slice on features must be removed when adding support to labels
    ds = tf.data.Dataset.from_tensor_slices( data )
    # TODO Create a function that generate masks and treat missing on tensorflow end
    if bool(opts.take_n):
      if cache_filepath: cache_filepath += '_take%d' % opts.take_n
      ds = ds.take( opts.take_n )
    if cache_filepath:
      if cache_filepath not in self._cached_filepath_dict:
        mkdir_p(cache_filepath)
        ds = ds.cache( cache_filepath )
        self._cached_filepath_dict[cache_filepath] = ds
      else:
        ds = ds.cache()
        print("Warning: Caching on memory although specified to cache on disk.\nReason: Dataset at '%s' is already currently being cached." % cache_filepath )
    if bool(opts.shuffle):
      ds = ds.shuffle(**opts.shuffle_kwargs)
    if opts.batch_size is not None:
      ds = ds.batch(opts.batch_size, drop_remainder = opts.drop_remainder)
    if opts.memory_cache:
      ds = ds.cache()
    #total_time = datetime.datetime.now() - start
    #print("Finished building dataset in %s." % total_time)
    return ds

  def _split_data(self, df, val_frac, test_frac, **split_kw ):
    squeeze = np.squeeze
    def split(X,y,frac,split_method,k,split_kw):
      if frac > 0.:
        if split_method.lower() == "kfold":
          if k is None:
            raise ValueError('Please specify kfold split')
          cv = sklearn.model_selection.StratifiedKFold(**split_kw) if split_kw.pop('stratified', y is not None) else sklearn.model_selection.KFold(**split_kw)
          gen = cv.split(X,y)
          in_idx, out_idx = next(itertools.islice(gen, k, None))
          ret = [X[in_idx], X[out_idx]]
          ret += [y[in_idx], y[out_idx]] if y is not None else [None, None]
          return ret
        elif split_method.lower() == "holdout":
          return sklearn.model_selection.train_test_split( X, y
            , test_size = frac
            , stratify = y if split_kw.pop('stratified', True if y is not None else False ) else None
            , **split_kw )
      return X, None, y, None
    def ensure_array(a):
      return np.array(a) if not isinstance(a, np.ndarray) else a
    import sklearn.model_selection
    # NOTE This function might fail if y is set to None
    X = np.expand_dims(df.to_numpy(), axis = 1)
    y = df[self._label] if self._label else None
    data_slices = split( X, y, test_frac
                       , split_kw.pop('test_split_method','holdout')
                       , split_kw.pop('test_k', None)
                       , split_kw = split_kw.pop('test_split_kw') )
    self.raw_test_data  = pd.DataFrame( squeeze( data_slices[1] ), columns = df.columns )
    self.raw_test_data  = self.raw_test_data.astype(df.dtypes.to_dict())
    self.raw_test_data  = self._parse_data( self.raw_test_data, df )
    data_slices = split( data_slices[0], data_slices[2], val_frac
                       , split_kw.pop('val_split_method','kfold')
                       , split_kw.pop('val_k', None) 
                       , split_kw = split_kw.pop('val_split_kw') )
    self.raw_val_data   = pd.DataFrame( squeeze( data_slices[1] ), columns = df.columns )
    self.raw_val_data   = self.raw_val_data.astype(df.dtypes.to_dict())
    self.raw_val_data   = self._parse_data( self.raw_val_data, df )
    self.raw_train_data = pd.DataFrame( squeeze( data_slices[0] ), columns = df.columns )
    self.raw_train_data = self.raw_train_data.astype(df.dtypes.to_dict())
    self.raw_train_data = self._parse_data( self.raw_train_data, df )
    return

  def _parse_data(self, df, orig):
    df.attrs = orig.attrs
    def parse_missing_categorical(df):
      mask = df != -1
      mask[mask==True] = 1.
      mask[mask==False] = 0.
      mask = mask.astype('float32')
      return {'data' : df, 'mask' : mask}
    def parse_missing_numerical(df):
      mask = df.notna()
      mask[mask==True] = 1.
      mask[mask==False] = 0.
      mask = mask.astype('float32')
      df[df.isna()] = 0.
      return {'data' : df, 'mask' : mask}
    ret = { 'categorical' : parse_missing_categorical(df[self.categorical_vars].astype('int32'))
          , 'binary'      : parse_missing_categorical(df[self.binary_vars].astype('int32'))
          , 'numerical'   : parse_missing_numerical(df[self.numerical_vars].astype('float32')) }
    return ret

