from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import sklearn
import datetime
import copy
import itertools

from ..misc import *
from .sampler_base import SamplerBase, SpecificFlowSamplingOpts

class _CacheStorage(CleareableCache):
  cached_functions = []

class ImagePathSampler(SamplerBase):

  def __init__(self, manager, **kw ):
    """
    TODO Help
    :param manager: a data manager instance 
    """
    super().__init__( raw_data = manager.data_path_list, **kw)
    self._pp = manager.pre_proc
    self._resize_to_resolution = retrieve_kw(kw, "resize_to_resolution", [128, 128] )
    del kw
    return

  @abstractmethod
  def retrieve_labels(self, X):
    # NOTE must return a list or an (n,1) nd.array
    pass

  def parse_input(self, input_dict):
    image = input_dict['image']
    image = tf.io.read_file(image)
    image = tf.image.decode_png(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, self._resize_to_resolution )
    input_dict['image'] = image
    input_dict['label'] = tf.cast( input_dict['label'], tf.uint8 )
    return input_dict

  def clear_cache(self):
    # FIXME
    _CacheStorage.clear_cached_functions()
    SamplerBase.clear_cache(self)

  def plot(self, model = None,  ds = "val", n_examples = 3):
    return

  def _make_dataset( self, data_dict, opts, cache_filepath ):
    #start = datetime.datetime.now()
    #print("Building new dataset...")
    # This will not work if attempting to forecast the past:
    try:
      sklearn.utils.validation.check_is_fitted( self._pp )
    except sklearn.exceptions.NotFittedError:
      self._pp.fit(self.raw_train_data)
    opts.set_unset_to_default( self, data_dict )
    # NOTE This slice on features must be removed when adding support to labels
    ds = tf.data.Dataset.from_tensor_slices( data_dict )
    ds = ds.map(self.parse_input) # NOTE flat_map is another option
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

  def _split_data(self, data_path_list, val_frac, test_frac, **split_kw ):
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
    X = np.expand_dims(np.array(data_path_list), axis = 1)
    y = ensure_array(self.retrieve_labels(data_path_list))
    data_slices = split( X, y, test_frac
                       , split_kw.pop('test_split_method','holdout')
                       , split_kw.pop('test_k', None)
                       , split_kw = split_kw.pop('test_split_kw') )
    self.raw_test_data = { 'image' : squeeze( data_slices[1] )
                         , 'label' : data_slices[3] }
    data_slices = split( data_slices[0], data_slices[2], val_frac
                       , split_kw.pop('val_split_method','kfold')
                       , split_kw.pop('val_k', None) 
                       , split_kw = split_kw.pop('val_split_kw') )
    self.raw_val_data = { 'image' : squeeze( data_slices[1] )
                        , 'label' : data_slices[3] }
    self.raw_train_data = { 'image' : squeeze( data_slices[0] )
                          , 'label' : data_slices[2] }
    return

