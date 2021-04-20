import functools
import os

class Container(object):
  pass

class CleareableCache(object):
  @classmethod
  def lru_cache(cl, *args, **kwargs):
    def decorator(func):
      func = functools.lru_cache(*args, **kwargs)(func)
      cl.cached_functions.append(func)
      return func
    return decorator

  @classmethod
  def cached_property(cl, *args, **kwargs):
    def decorator(func):
      func = functools.lru_cache(*args, **kwargs)(func)
      cl.cached_functions.append(func)
      return property(func)
    return decorator

  @classmethod
  def clear_cached_functions(cl):
    for func in cl.cached_functions:
      func.cache_clear()


class NotSetType( type ):
  def __bool__(self):
    return False
  def __len__(self):
    return False
  __nonzero__ = __bool__
  def __repr__(self):
    return "<+NotSet+>"
  def __str__(self):
    return "<+NotSet+>"

class NotSet( object, metaclass=NotSetType ): 
  """As None, but can be used with retrieve_kw to have a unique default value
  through all job hierarchy."""
  pass

def retrieve_kw( kw, key, default = NotSet ):
  """
  Use together with NotSet to have only one default value for your job
  properties.
  """
  if not key in kw or kw[key] is NotSet:
    kw[key] = default
  return kw.pop(key)

class Iterable(object):
  def __enter__(self):
    pass

  def __exit__(self, exc_type, exc_value, traceback):
    if exc_type is None:
      to_delete = [k for k in self.__dict__ if k.startswith('_l_')]
      for d in to_delete: 
        del self.__dict__[d]

def fix_model_layers(model):
  from tensorflow.keras.layers import Layer
  model._layers = [
    layer for layer in model._layers if isinstance(layer, Layer)
  ]
  return model

def mkdir_p(path):
  path = os.path.expandvars( path )
  if not os.path.exists( path ):
    os.makedirs(path)

def to_tuple(i):
  if not isinstance(i, tuple):
    if type(i) is list:
      return tuple(i)
    else:
      return (i,)
  return i

import signal
import logging

class DelayedKeyboardInterrupt(object):
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logging.debug('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)
