from .meter_base import ScalarEff, GenerativeEffMeter
from ..misc import *

import tensorflow as tf
import numpy as np

class ePDF(ScalarEff,GenerativeEffMeter):

  def __init__(self, name = "ePDF", range_value = None):
    super().__init__(name)
    self.range_value = range_value

  def initialize(self, x_g1_list, x_g2 = None, xmask_g1 = None, xmask_g2 = None):
    if self.initialized:
      return
    if xmask_g1 is not None or xmask_g2 is not None:
      raise NotImplementedError("ePDF is not currently implemented for masked data")
    self.reset()
    self.edges = []
    self.dfR = []
    
    self.initialized = True
    
    # for x_g1 in x_g1_list:
    #   self.xs = x_g1.shape
    #   # NOTE dfR and dfG are currently flattened
    #   for t in range(self.xs[1]):
    #     for d in range(self.xs[2]):
    #       dfR, edges = np.histogram(x_g1[:,t,d], bins='auto', density=True, range=self.range_value[t][d])
    #       self.edges.append(edges)
    #       self.dfR.append(dfR)
          
    
    self.xs = x_g1_list[0].shape
    # NOTE dfR and dfG are currently flattened
    for t in range(self.xs[1]):
      for d in range(self.xs[2]):
        lx_g1=[]
        for x_g1 in x_g1_list:
          lx_g1.append(x_g1[:,t,d])
        dfR, edges = np.histogram(lx_g1, bins='auto', density=True, range=self.range_value[t][d])
        self.edges.append(edges)
        self.dfR.append(dfR)

  def accumulate(self, x_gen_list, xmask = None ):
    if xmask is not None:
      raise NotImplementedError("ePDF is not currently implemented for masked data")
    # if self.i > 0:
    #   raise NotImplementedError("ePDF is not able to work with multiple minibatches")
    # TODO How to save the histogram through time?
    # "Custom scalar"? Customize histogram?

    self.start
    
    # for x_gen in x_gen_list:
    #   eiter = iter(self.edges)
    #   for t in range(self.xs[1]):
    #     for d in range(self.xs[2]):
    #       dfG = np.histogram(x_gen[:,t,d], bins=next(eiter), density=True)[0]
    #       self.dfG.append(dfG)
    #   self.i += 1
    # self.stop
    
    eiter = iter(self.edges)
    for t in range(self.xs[1]):
      for d in range(self.xs[2]):
        lx_gen = []
        for x_gen in x_gen_list:
          lx_gen.append(x_gen[:,t,d])
        dfG = np.histogram(x_gen[:,t,d], bins=next(eiter), density=True)[0]
        self.dfG.append(dfG)
        
    self.stop

  def retrieve(self):
    # print("xs:", self.xs)
    # print("edges:", self.edges)
    # print("dfR:", self.dfR)
    # print("dfG:", self.dfG)
    self.start
    self.ePDF = 0.
    for dfR, dfG in zip(self.dfR, self.dfG):
      self.ePDF += tf.reduce_mean( tf.math.abs( dfR - dfG ) )
    self.ePDF /= len(self.dfR)
    self.stop
    self.print
    return self.ePDF

  def reset(self):
    super().reset()
    self.ePDF = 0.
    self.i = 0
    self.dfG = []
    self.initialized = False
