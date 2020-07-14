class NotSetType( type ):
  def __bool__(self):
    return False
  __nonzero__ = __bool__
  def __repr__(self):
    return "<+NotSet+>"
  def __str__(self):
    return "<+NotSet+>"

class NotSet( object ): 
  """As None, but can be used with retrieve_kw to have a unique default value
  through all job hierarchy."""
  __metaclass__ = NotSetType

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
