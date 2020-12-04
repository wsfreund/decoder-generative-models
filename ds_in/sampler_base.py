import numpy as np
import tensorflow as tf
import copy

try:
  from misc import *
except ImportError:
  from .misc import *

# FIXME This is a singleton shared within all instances
# TODO Implement a dict where the key is each instance
class _CacheStorage(CleareableCache):
  cached_functions = []

class SamplerBase(object):

  def __init__(self, manager, **kw ):
    self._val_frac                   = retrieve_kw(kw, "val_frac",                   .2                                   )
    self._test_frac                  = retrieve_kw(kw, "test_frac",                  .2                                   )
    self._shuffle                    = retrieve_kw(kw, "shuffle",                    True                                 )
    self._shuffle_kwargs             = retrieve_kw(kw, "shuffle_kwargs",             {}                                   )
    self._batch_size                 = retrieve_kw(kw, "batch_size",                 128                                  )
    if not "reshuffle_each_iteration" in self._shuffle_kwargs:
      self._shuffle_kwargs["reshuffle_each_iteration"] = False
    self._pp = manager.pre_proc
    self._split_data( manager.df, self._val_frac, self._test_frac )

  @_CacheStorage.cached_property()
  def sampler_from_train_ds(self):
    return self._make_dataset(self.train_df)

  @_CacheStorage.cached_property()
  def sampler_from_val_ds(self):
    return self._make_dataset(self.val_df)

  @_CacheStorage.cached_property()
  def sampler_from_test_ds(self):
    return self._make_dataset(self.test_df)

  def sample(self, ds = "val"):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    # No example batch was found, so get one from the `.train` dataset
    if ds == "train":
      f_iter = self._single_sample_cached_train_iter
    elif ds == "val":
      f_iter = self._single_sample_cached_val_iter
    elif ds == "test":
      f_iter = self._single_sample_cached_test_iter
    else:
      raise RuntimeError("unknown dataset %s.", ds)
    try:
      result = next(f_iter)
    except StopIteration:
      f.cache_clear()
      result = next(f_iter)
    return result

  @_CacheStorage.cached_property()
  def _single_sample_cached_train_iter(self):
    return iter(self._make_dataset(self.train_df,batch=False))

  @_CacheStorage.cached_property()
  def _single_sample_cached_val_iter(self):
    return iter(self._make_dataset(self.val_df,batch=False))

  @_CacheStorage.cached_property()
  def _single_sample_cached_test_iter(self):
    return iter(self._make_dataset(self.test_df,batch=False))

  def _make_dataset(self, df, batch = True):
    raise NotImplementedError("Overload _make_dataset for class %s", self.__class__.__name__)

  def _split_data(self, df, val_frac, test_frac ):
    # TODO Add compatibility with k-fold or bootstapping 
    raise NotImplementedError("Overload _split_data for class %s", self.__class__.__name__)
