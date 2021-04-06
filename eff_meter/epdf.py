from .meter_base import GenerativeEffMeter
from ..misc import *

import tensorflow as tf
import numpy as np
import itertools

class ePDFMeter(GenerativeEffMeter):

  def __init__(self, range_value, name = "ePDF", data_parser = lambda x: x
              , gen_parser = lambda x: x, nbins = 50):
    super().__init__(name, data_parser, gen_parser)
    if not isinstance(range_value, np.ndarray):
      range_value = np.ndarray(range_value)
    self.range_value = range_value
    self.nbins = nbins
    self.xs = None
    self.dfR = None
    self.dfG = None

  def update_on_parsed_data(self, data, mask):
    if mask is not None:
      raise NotImplementedError("%s is not currently implemented for masked data" % self.__class__.__name__)
    if self.xs is None:
      self.xs = data.shape
      self.dfR = np.ndarray(self.xs[1:], dtype=np.object)
      self.dfG = np.ndarray(self.xs[1:], dtype=np.object)
    for indexes in itertools.product(*map(range,self.xs[1:])):
      dfR, _ = np.histogram(data[(slice(None),)+indexes].numpy(), bins=self.nbins, density=True, range=self.range_value[indexes])
      if self.dfR[indexes] is not None:
        # TODO Check dfR update
        self.dfR[indexes] += dfR
      else:
        self.dfR[indexes] = dfR

  def update_on_parsed_gen(self, data, mask = None ):
    if mask is not None:
      raise NotImplementedError("%s is not currently implemented for masked data" % self.__class__.__name__)
    for indexes in itertools.product(*map(range,self.xs[1:])):
      dfG, _ = np.histogram(data[(slice(None),)+indexes].numpy(), bins=self.nbins, density=True, range=self.range_value[indexes])
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
    self.dfG = np.ndarray(self.xs[1:], dtype=np.object)
