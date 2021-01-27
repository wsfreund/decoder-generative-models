from abc import ABC, abstractmethod
import tensorflow as tf
import datetime

class EffMeterBase(ABC):

  def __init__(self, name):
    self.name = name
    self._initialized = False
    self._deltatime = datetime.timedelta()

  @abstractmethod
  def retrieve(self):
    pass

  @abstractmethod
  def to_summary(self):
    pass

  @property
  def initialized(self):
    return self._initialized

  @initialized.setter
  def initialized(self, val):
    self._initialized = val

  @property
  def start(self):
    self._ctime = datetime.datetime.now()

  @property
  def stop(self):
    self._deltatime +=  datetime.datetime.now() - self._ctime

  def reset(self):
    self._deltatime = datetime.timedelta()

  @property
  def print(self):
    print("%s computation time: %s" % (self.name, self._deltatime))

class GenerativeEffMeter(EffMeterBase):

  @abstractmethod
  def initialize(self, x_data, x_mask ):
    pass

  @abstractmethod
  def accumulate(self, x_gen, x_mask ):
    pass

class ModelEffMeter(EffMeterBase):

  def __init__(self, name, model = None):
    super().__init__(name)
    self.model = model

  @property
  def initialized(self):
    return self._initialized if self.model else False

  @initialized.setter
  def initialized(self, val):
    if val == True and self.model is None:
      raise RuntimeError("Attempted to initialize meter without a model!")
    self._initialized = val

class ScalarEff(object):
  def to_summary(self):
    eff = self.retrieve()
    tf.summary.scalar(self.name, eff)

class HistogramEff(object):
  def to_summary(self):
    hist = self.retrieve()
    tf.summary.histogram(self.name, hist)

class ScalaAndHistogramEff(object):
  def to_summary(self):
    eff, hist = self.retrieve()
    # FIXME
    tf.summary.scalar(self.name, eff)
    tf.summary.histogram(self.name, hist)
