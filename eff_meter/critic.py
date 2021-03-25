from .meter_base import ScalarEff, GenerativeEffMeter, ModelEffMeter
from ..misc import *

import tensorflow as tf
import numpy as np

class CriticEffMeter(ScalarEff, GenerativeEffMeter, ModelEffMeter):

  def __init__(self, name = "critic"):
    super().__init__(name)

  def initialize(self, x_data_list, x_gen = None, xmask_g1 = None, xmask_g2 = None):
    if self.initialized:
      return
    if xmask_g1 is not None or xmask_g2 is not None:
      raise NotImplementedError("ePDF is not currently implemented for masked data")
    if not isinstance(x_data_list, list):
      x_data_list = [x_data_list]
      
    self.reset()
    self.initialized = True
    
    self.avg_output_data = tf.reduce_mean( self.model(x_data_list[0]) )
    for xdata in x_data_list[1:]:
      self.avg_output_data = tf.math.add(self.avg_output_data, tf.reduce_mean( self.model(xdata) ))
    

  def accumulate(self, x_gen_list, xmask = None ):
    if xmask is not None:
      raise NotImplementedError("ACF is not currently implemented for masked data")
    if not isinstance(x_gen_list, list):
      x_gen_list = [x_gen_list]
      
    self.start
    
    self.avg_output_gen = tf.reduce_mean( self.model(x_gen_list[0]) )
    self.i += 1
    for xgen in x_gen_list[1:]:
      self.avg_output_gen = tf.math.add(self.avg_output_gen, tf.reduce_mean( self.model(xgen) ))
      self.i += 1
      
    self.stop
    return self.avg_output_gen

  def retrieve(self, gen = False):
    if not gen: self.print
    return (self.avg_output_gen/self.i) if gen else (self.avg_output_data/self.i)

  def reset(self):
    self.avg_output_data = 0.
    self.i = 0
    self.initialized = False

  def to_summary(self, gen = False):
    eff = self.retrieve( gen )
    tf.summary.scalar(self.name, eff)
