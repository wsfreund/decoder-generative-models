from .meter_base import EffMeterBase
from ..misc import *

import tensorflow as tf
import numpy as np

class AE_EffMeter(EffMeterBase):
  def __init__(self, name = "ae_meter", data_parser = lambda x: x):
    super().__init__(name, data_parser)
    self.data_loss_dict = {}

  def initialize(self, wrapper):
    EffMeterBase.initialize(self,wrapper)
    self._training_model = wrapper._training_model
    self._compute_surrogate_loss = wrapper._compute_surrogate_loss
    prev_data_parser = self.data_parser
    self.data_parser = lambda x: self._compute_loss(prev_data_parser(x))

  def _compute_loss(self, x):
    x_reco = self._training_model( x )
    ae_loss_dict = self._compute_surrogate_loss( x, x_reco )
    return ae_loss_dict

  def update_on_parsed_data(self, loss_dict, mask = None, corr_new = 1.,  corr_tot = 1.):
    if mask is not None:
      raise NotImplementedError("%s is not currently implemented for masked data" % self.__class__.__name__)
    for k, v in loss_dict.items():
      if k in self.data_loss_dict:
        self.data_loss_dict[k] += v
      else:
        self.data_loss_dict[k] = v
    return

  def retrieve(self):
    ret = {k : (v/self.data_batch_counter if self.data_batch_counter else v) for k, v in self.data_loss_dict.items()}
    return ret

  def reset(self):
    super().reset()
    self.data_loss_dict = {}

