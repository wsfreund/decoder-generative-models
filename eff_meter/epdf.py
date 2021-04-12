from .meter_base import GenerativeEffMeter, GenerativeEffBufferedMeter
from ..misc import *

import tensorflow as tf
import numpy as np
import itertools

class ePDFMeter(GenerativeEffMeter):

  def __init__(self, range_value, name = "ePDF", data_parser = lambda x: x
              , gen_parser = lambda x: x, nbins = 50, **kw):
    super().__init__(name = name, data_parser = data_parser, gen_parser = gen_parser, **kw)
    if not isinstance(range_value, np.ndarray):
      range_value = np.ndarray(range_value)
    self.range_value = range_value
    self.nbins = nbins
    self.xs = self.range_value.shape[:-1]
    self.dfR = np.ndarray(self.xs, dtype=np.object)
    self.dfG = np.ndarray(self.xs, dtype=np.object)

  def update_on_parsed_data(self, data, mask):
    if mask is not None:
      raise NotImplementedError("%s is not currently implemented for masked data" % self.__class__.__name__)
    if not isinstance(data, np.ndarray):
      data = data.numpy()
    for indexes in itertools.product(*map(range,self.xs)):
      dfR = np.histogram(data[(slice(None),)+indexes], bins=self.nbins, density=True, range=self.range_value[indexes])[0]
      if self.dfR[indexes] is not None:
        self.dfR[indexes] += dfR
      else:
        self.dfR[indexes] = dfR

  def update_on_parsed_gen(self, data, mask = None ):
    if mask is not None:
      raise NotImplementedError("%s is not currently implemented for masked data" % self.__class__.__name__)
    if not isinstance(data, np.ndarray):
      data = data.numpy()
    for indexes in itertools.product(*map(range,self.xs)):
      dfG = np.histogram(data[(slice(None),)+indexes], bins=self.nbins, density=True, range=self.range_value[indexes])[0]
      if self.dfG[indexes] is not None:
        self.dfG[indexes] += dfG
      else:
        self.dfG[indexes] = dfG

  def retrieve(self):
    if not self._locked_data_statistics:
      for indexes in itertools.product(*map(range,self.dfR.shape)):
        self.dfR[indexes] /= self.data_batch_counter
      self._locked_data_statistics = True
    for indexes in itertools.product(*map(range,self.dfG.shape)):
      self.dfG[indexes] /= self.gen_batch_counter
    ePDF = 0.
    self.ePDF_per_feature = np.zeros( self.dfR.shape, dtype = np.float32 )
    for indexes in itertools.product(*map(range,self.dfR.shape)):
      result = np.mean( np.abs( self.dfR[indexes] - self.dfG[indexes] ) )
      self.ePDF_per_feature[indexes] = result
      ePDF += result
      # TODO  keep record of the reduced statistics
    ePDF /= np.prod(self.dfR.shape)
    return { self.name : ePDF }

  def reset(self):
    super().reset()
    self.dfG = np.ndarray(self.xs, dtype=np.object)

class ePDFBufferedMeter(ePDFMeter, GenerativeEffBufferedMeter):

  def __init__( self, range_value, name = "ePDF", data_parser = lambda x: x, gen_parser = lambda x: x, nbins = 50
              , data_buffer = None, gen_buffer = None, max_buffer_size = 16, **kw):
    super().__init__( range_value = range_value, name = name, data_parser = data_parser, gen_parser = gen_parser, nbins = nbins
                    , data_buffer = data_buffer, gen_buffer = gen_buffer, max_buffer_size = max_buffer_size, **kw)
