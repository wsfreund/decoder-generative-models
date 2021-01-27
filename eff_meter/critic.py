from .meter_base import ScalarEff, GenerativeEffMeter, ModelEffMeter
from ..misc import *

import tensorflow as tf
import numpy as np

class CriticEffMeter(ScalarEff, GenerativeEffMeter, ModelEffMeter):

  def __init__(self, name = "critic"):
    super().__init__(name)

  def initialize(self, x_data, x_gen = None, xmask_g1 = None, xmask_g2 = None):
    if self.initialized:
      return
    if xmask_g1 is not None or xmask_g2 is not None:
      raise NotImplementedError("ePDF is not currently implemented for masked data")
    self.reset()
    self.x_data = x_data
    self.initialized = True

  def accumulate(self, x_gen, xmask = None ):
    if xmask is not None:
      raise NotImplementedError("ACF is not currently implemented for masked data")
    if self.i > 0:
      raise NotImplementedError("ACF is not able to work with multiple minibatches")
    self.start
    self.avg_output_data = tf.reduce_mean( self.model(self.x_data) )
    self.avg_output_gen = tf.reduce_mean( self.model(x_gen) )
    self.stop
    return self.avg_output_data

  def retrieve(self, gen = False):
    if not gen: self.print
    return self.avg_output_gen if gen else self.avg_output_data

  def reset(self):
    self.avg_output_data = 0.
    self.i = 0

  def to_summary(self, gen = False):
    eff = self.retrieve( gen )
    tf.summary.scalar(self.name, eff)
