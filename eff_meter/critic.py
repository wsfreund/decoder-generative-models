from .meter_base import EffMeterBase, GenerativeEffMeter
from ..misc import *

import tensorflow as tf
import numpy as np

class CriticEffMeter(GenerativeEffMeter):
  def __init__(self, name = "critic", data_parser = lambda x: x):
    super().__init__(name, data_parser)
    self.avg_output_data = 0.
    self.avg_output_gen = 0.


  def initialize(self, wrapper):
    self.data_parser = lambda x: wrapper.critic(x)
    self.gen_parser = lambda x: wrapper.critic(x)

  def update_on_parsed_data(self, data, mask = None):
    if mask is not None:
      raise NotImplementedError("%s is not currently implemented for masked data" % self.__class__.__name__)
    self.avg_output_data += tf.reduce_mean( data )

  def update_on_parsed_gen(self, data, mask = None ):
    if mask is not None:
      raise NotImplementedError("%s is not currently implemented for masked data" % self.__class__.__name__)
    self.avg_output_gen += tf.reduce_mean( data )

  def retrieve(self):
    self.avg_output_data /= self.data_batch_counter
    self.avg_output_gen /= self.gen_batch_counter
    return { self.name + '_data' : self.avg_output_data
           , self.name + '_gen' : self.avg_output_gen
           , self.name + '_delta' : self.avg_output_gen - self.avg_output_data}

  def reset(self):
    super().reset()
    self.avg_output_data = 0.
    self.avg_output_gen = 0.

