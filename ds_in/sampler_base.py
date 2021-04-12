from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import copy
import itertools

from ..misc import *

class SpecificFlowSamplingOpts(object):

  def __init__( self, batch_size
              , take_n = NotSet, drop_remainder = NotSet
              , shuffle = NotSet, shuffle_kwargs = NotSet
              , memory_cache = False ):
    assert batch_size is not NotSet
    self.batch_size     = batch_size
    self.take_n         = take_n
    self.drop_remainder = drop_remainder
    self.shuffle        = shuffle
    self.shuffle_kwargs = shuffle_kwargs
    self.memory_cache   = memory_cache

  def set_unset_to_default(self, sampler, df):
    if self.take_n is NotSet:
      self.take_n = None
    if self.drop_remainder is NotSet:
      self.drop_remainder = True
    if self.shuffle is NotSet:
      if self.batch_size is not None:
        if self.take_n not in (None, NotSet):
          n_examples = self.take_n
        else:
          n_examples, n_dims = df.shape
        self.shuffle = True if self.batch_size < n_examples // 4 else False
      else:
        self.shuffle = True
    if self.shuffle_kwargs is NotSet:
      self.shuffle_kwargs = {}
    if "reshuffle_each_iteration" not in self.shuffle_kwargs:
      self.shuffle_kwargs["reshuffle_each_iteration"] = False
    if "buffer_size" not in self.shuffle_kwargs:
      if self.batch_size is not None:
        self.shuffle_kwargs["buffer_size"] = self.batch_size * 8

# FIXME This is a singleton shared within all instances
# TODO Implement a dict where the key is each instance
class _CacheStorage(CleareableCache):
  cached_functions = []

class SamplerBase(ABC):
  """
  NOTE: Any modification to sampler object requires removing any previous
  cache_filepath to take effect.
  """

  def __init__( self, raw_data
              , split_kw = {}, **kw ):
    """
    There are several sampler functions to be used:
    - training_sampler: Iteration considering the training conditions. These
      conditions are provided by training_sampler_kwargs. The arguments are
      provided to make_dataset.
    - evaluation_sampler: Iteration using evaluation conditions. These
      conditions are provided by eval_sampler_kwargs. The arguments are
      provided to make_dataset.
    - new sampler: Create a new sampler using custom conditions.
    """
    # Splitting instructions
    self._val_frac                   = retrieve_kw(kw, "val_frac",  .2 )
    self._test_frac                  = retrieve_kw(kw, "test_frac", .2 )
    self._split_data( raw_data, self._val_frac, self._test_frac, **split_kw )
    # Specify specific sampling instructions for each dataset
    SpecificOptsCls = retrieve_kw(kw, "specific_flow_sampling_opt_class", SpecificFlowSamplingOpts )
    training_sampler_kwargs  = retrieve_kw(kw, "training_sampler_kwargs", {} )
    if not "batch_size" in training_sampler_kwargs:
      training_sampler_kwargs["batch_size"] = 128
    if not "drop_remainder" in training_sampler_kwargs:
      training_sampler_kwargs["drop_remainder"] = True
    self.training_sampler_opts = SpecificOptsCls( **training_sampler_kwargs )
    eval_sampler_kwargs = retrieve_kw(kw, "eval_sampler_kwargs", {} )
    if "batch_size" not in eval_sampler_kwargs:
      eval_sampler_kwargs["batch_size"] = self.training_sampler_opts.batch_size * 16
    if not "drop_remainder" in eval_sampler_kwargs:
      eval_sampler_kwargs["drop_remainder"] = False
    if "take_n" not in eval_sampler_kwargs:
      eval_sampler_kwargs["take_n"] = 2**20 # 1,048,576
    self.eval_sampler_opts = SpecificOptsCls( **eval_sampler_kwargs )
    # Other configuration
    self._cache_filepath = retrieve_kw(kw, "cache_filepath", '' )
    self._cached_filepath_dict = {}

  def new_sampler_from_train_ds(self, **kw):
    """
    Keywords are passed to make_dataset
    """
    return self._make_dataset(self.raw_train_data, **kw)

  def new_sampler_from_val_ds(self, **kw):
    """
    Keywords are passed to make_dataset
    """
    return self._make_dataset(self.raw_val_data, **kw)

  def new_sampler_from_test_ds(self, **kw):
    """
    Keywords are passed to make_dataset
    """
    return self._make_dataset(self.raw_test_data, **kw)

  @property
  def has_train_ds(self):
    return hasattr(self,"raw_train_data")

  @property
  def has_val_ds(self):
    return hasattr(self,"raw_val_data")

  @property
  def has_test_ds(self):
    return hasattr(self,"raw_test_data")
 
  @abstractmethod
  def _make_dataset( self, df, opts, cache_filepath):
    pass

  @abstractmethod
  def _split_data(self, val_frac, test_frac ):
    pass

  @_CacheStorage.cached_property()
  def training_sampler(self):
    """
    Sampler on the same conditions as those specified for class instance
    """
    cache_filepath = self._cache_filepath
    label = 'train_surrogate'
    if cache_filepath: cache_filepath += '_' + label
    ds = self._make_dataset(self.raw_train_data, self.training_sampler_opts, cache_filepath )
    ds.label = label
    ds.opts = self.training_sampler_opts
    return ds

  def evaluation_sampler_from_ds(self, ds):
    if ds == "train":
      return self.evaluation_sampler_from_train_ds()
    elif ds == "val":
      return self.evaluation_sampler_from_val_ds()
    elif ds == "test":
      return self.evaluation_sampler_from_test_ds()
    else:
      raise RuntimeError("Unspecified dataset label %s" % ds)

  @_CacheStorage.cached_property()
  def evaluation_sampler_from_train_ds(self):
    """
    Sampler on the same conditions as those specified for class instance
    """
    cache_filepath = self._cache_filepath
    label = 'train_perf'
    if cache_filepath: cache_filepath += '_' + label
    ds = self._make_dataset(self.raw_train_data, self.eval_sampler_opts, cache_filepath )
    ds.label = label
    ds.opts = self.eval_sampler_opts
    return ds

  @_CacheStorage.cached_property()
  def evaluation_sampler_from_val_ds(self):
    """
    Sampler on the same conditions as those specified for class instance
    """
    cache_filepath = self._cache_filepath
    label = 'val_perf'
    if cache_filepath: cache_filepath += '_' + label
    ds = self._make_dataset(self.raw_val_data, self.eval_sampler_opts, cache_filepath )
    ds.label = label
    ds.opts = self.eval_sampler_opts
    return ds

  @_CacheStorage.cached_property()
  def evaluation_sampler_from_test_ds(self):
    """
    Sampler on the same conditions as those specified for class instance
    """
    cache_filepath = self._cache_filepath
    label = 'test_perf'
    if cache_filepath: cache_filepath += '_' + label
    ds = self._make_dataset(self.raw_test_data, self.eval_sampler_opts, cache_filepath )
    ds.label = label
    ds.opts = self.eval_sampler_opts
    return ds

  def batch_subsample(self, n_samples = 1, mode = "first_n", ds = "val"):
    """
    Sample data using evaluation iteration condition with minibatch. Extra
    minibatch samples are discarded.

    mode can take the following values:
    - first_n: sample the first n values from each batch;
    """
    if ds == "train":
      f_iter = self._batch_sample_cached_train_iter
      fget = type(self)._batch_sample_cached_train_iter.fget
    elif ds == "val":
      f_iter = self._batch_sample_cached_val_iter
      fget = type(self)._batch_sample_cached_val_iter.fget
    elif ds == "test":
      f_iter = self._batch_sample_cached_test_iter
      fget = type(self)._batch_sample_cached_test_iter.fget
    else:
      raise RuntimeError("unknown dataset %s.", ds)
    try:
      samples = next(f_iter)
    except StopIteration:
      # uncache and cache new iterator
      fget.cache_clear()
      f_iter = fget.__get__(self, type(self))()
      samples = next(f_iter)
    if mode == "first_n":
      if isinstance(samples, dict):
        ret_samples = {}
        for k, v in samples.items():
          if isinstance(v,(tuple,list)):
            ret_samples[k] = [v2[:n_samples] for v2 in v]
          else:
            ret_samples[k] = v[:n_samples]
        return ret_samples
      elif isinstance(samples,(tuple,list)):
        return [v[:n_samples] for v in samples]
      else:
        return samples[:n_samples]
    else:
      raise ValueError("invalid mode '%s'" % mode)

  def sample(self, n_samples = 1, ds = "val"):
    """
    Sample data using evaluation iteration conditions, but without using
    minibatch.  It will create a new cached sampler, which may require creating
    a new shuffle buffer.
    """
    if ds == "train":
      f_iter = self._single_sample_cached_train_iter
      fget = type(self)._single_sample_cached_train_iter.fget
    elif ds == "val":
      f_iter = self._single_sample_cached_val_iter
      fget = type(self)._single_sample_cached_val_iter.fget
    elif ds == "test":
      f_iter = self._single_sample_cached_test_iter
      fget = type(self)._single_sample_cached_test_iter.fget
    else:
      raise RuntimeError("unknown dataset %s.", ds)
    samples = []
    for _ in range(n_samples):
      try:
        sample = next(f_iter)
      except StopIteration:
        # uncache and cache new iterator
        fget.cache_clear()
        f_iter = fget.__get__(self, type(self))()
        sample = next(f_iter)
      samples.append(sample)
    if n_samples > 1:
      if isinstance(sample, dict):
        ret_samples = {}
        for k in sample.keys():
          if isinstance(sample[k],(tuple,list)):
            ret_samples[k] = [tf.stack([s[k][i] for s in samples]) for i in range(len(sample[k]))]
          else:
            ret_samples[k] = tf.stack([s[k] for s in samples])
        samples = ret_samples
      elif isinstance(sample, (tuple,list)):
        samples = [tf.stack([s[i] for s in samples]) for i in range(len(sample))]
      else:
        samples = tf.stack(samples)
    else:
      samples = samples[0]
    return samples

  @_CacheStorage.cached_property()
  def _batch_sample_cached_train_iter(self):
    return iter(self.evaluation_sampler_from_train_ds)

  @_CacheStorage.cached_property()
  def _batch_sample_cached_val_iter(self):
    return iter(self.evaluation_sampler_from_val_ds)

  @_CacheStorage.cached_property()
  def _batch_sample_cached_test_iter(self):
    return iter(self.evaluation_sampler_from_test_ds)

  @_CacheStorage.cached_property()
  def _single_sample_cached_train_iter(self):
    eval_opts = copy.copy(self.eval_sampler_opts)
    eval_opts.batch_size = None
    eval_opts.take_n = None
    eval_opts.shuffle = True
    eval_opts.shuffle_kwargs = dict( buffer_size = self.eval_sampler_opts.batch_size*8 )
    eval_opts.cache = False
    cache_filepath = ''
    return iter(self._make_dataset(self.raw_train_data,eval_opts,cache_filepath))

  @_CacheStorage.cached_property()
  def _single_sample_cached_val_iter(self):
    eval_opts = copy.copy(self.eval_sampler_opts)
    eval_opts.batch_size = None
    eval_opts.take_n = None
    eval_opts.shuffle = True
    eval_opts.shuffle_kwargs = dict( buffer_size = self.eval_sampler_opts.batch_size*8 )
    eval_opts.cache = False
    cache_filepath = ''
    return iter(self._make_dataset(self.raw_val_data,eval_opts,cache_filepath))

  @_CacheStorage.cached_property()
  def _single_sample_cached_test_iter(self):
    eval_opts = copy.copy(self.eval_sampler_opts)
    eval_opts.batch_size = None
    eval_opts.take_n = None
    eval_opts.shuffle = True
    eval_opts.shuffle_kwargs = dict( buffer_size = self.eval_sampler_opts.batch_size*8 )
    eval_opts.cache = False
    cache_filepath = ''
    return iter(self._make_dataset(self.raw_test_data,eval_opts,cache_filepath))
