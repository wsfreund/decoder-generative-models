from .meter_base import GenerativeEffMeter, GenerativeEffBufferedMeter
from ..misc import *

import tensorflow as tf
import numpy as np
import itertools

class GenerativePerfSamplerCounter(GenerativeEffMeter):

  def __init__(self, name = "PerfSampleCounter", data_parser = lambda x: x
              , gen_parser = lambda x: x, **kw):
    super().__init__(name = name, data_parser = data_parser, gen_parser = gen_parser, **kw)
    self.n_data_samp = 0
    self.n_gen_samp = 0

  def update_on_parsed_data(self, data, mask, corr_new = 1.,  corr_tot = 1.):
    if mask is not None:
      raise NotImplementedError("%s is not currently implemented for masked data" % self.__class__.__name__)
    self.n_data_samp += data.shape[0]

  def update_on_parsed_gen(self, data, mask = None, corr_new = 1.,  corr_tot = 1. ):
    if mask is not None:
      raise NotImplementedError("%s is not currently implemented for masked data" % self.__class__.__name__)
    self.n_gen_samp += data.shape[0]

  def retrieve(self):
    if not self._locked_data_statistics:
      self._locked_data_statistics = True
    return { self.name + '_data' : self.n_data_samp
           , self.name + '_data_batches' : self.data_batch_counter
           , self.name + '_gen' : self.n_gen_samp 
           , self.name + '_gen_batches' : self.gen_batch_counter }

  def reset(self):
    super().reset()
    self.n_gen_samp = 0

