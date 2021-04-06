from .meter_base import GenerativeEffMeter
from ..misc import *

import tensorflow as tf

class ACFMeter(GenerativeEffMeter):

  def __init__(self, name = "ACF", data_parser = lambda x: x, gen_parser = lambda x: x, start_lag = 1, stop_lag = 2):
    super().__init__(name, data_parser, gen_parser)
    self.start_lag               = start_lag
    self.stop_lag                = stop_lag
    self.xreal_mean_acc          = tf.Variable(0., dtype = tf.float32)
    self.xreal_meansquare_acc    = tf.Variable(0., dtype = tf.float32)
    self.xreal_mean_delta_t_prod = tf.Variable([0.]*(stop_lag-start_lag), dtype = tf.float32)
    self.xgen_mean_acc           = tf.Variable(0., dtype = tf.float32)
    self.xgen_meansquare_acc     = tf.Variable(0., dtype = tf.float32)
    self.xgen_mean_delta_t_prod  = tf.Variable([0.]*(stop_lag-start_lag), dtype = tf.float32)

  def update_on_parsed_data(self, data, mask = None):
    if mask is not None:
      raise NotImplementedError("%s is not currently implemented for masked data" % self.__class__.__name__)
    accumulate_acf_ingredients( data, self.xreal_mean_acc, self.xreal_meansquare_acc, self.xreal_mean_delta_t_prod )

  def update_on_parsed_gen(self, data, mask = None):
    if mask is not None:
      raise NotImplementedError("%s is not currently implemented for masked data" % self.__class__.__name__)
    accumulate_acf_ingredients( data, self.xgen_mean_acc, self.xgen_meansquare_acc, self.xgen_mean_delta_t_prod )

  @tf.function
  def _compute_stats(self, acf_data, acf_gen ):
    # NOTE Norm-2 is employed in the original code although norm-1 is mentioned
    # on the paper
    return tf.linalg.norm(tf.math.divide(tf.math.subtract( self.acf_gen, self.acf_data), self.i), ord = 2, axis = 0 )

  def _compute_acf(self, mean_acc, meansquare_acc, mean_delta_t_prod, counter ):
    mean              = mean_acc / counter
    mean_square       = mean_square_acc / counter
    mean_delta_t_prod = mean_delta_t_prod / counter
    square_mean       = tf.math.square( mean )
    acf = tf.math.divide( tf.math.subtract( mean_delta_t_prod, square_mean )
                        , tf.math.subtract( xreal_meansquare, square_mean ) )
    return acf


  def retrieve(self):
    self.start
    if not self._locked_data_statistics:
      self.acf_data = self._compute_acf( self.xreal_mean_acc, self.xreal_meansquare_acc, self.xreal_mean_delta_t_prod, self.data_batch_counter )
      self._locked_data_statistics = True
    self.acf_gen = self._compute_acf( self.xgen_mean_acc, self.xgen_meansquare_acc, self.xgen_mean_delta_t_prod, self.data_batch_counter )
    total_diff = self._compute_stats( self.acf_data, self.acf_gen )
    self.stop
    return { self.name : total_diff }

  def reset(self):
    super().reset()
    self.xgen_mean_acc.assign(0.)
    self.xgen_meansquare_acc.assign(0.)
    self.xgen_mean_delta_t_prod  = tf.zeros_like(self.xgen_mean_delta_t_prod)

@tf.function
def accumulate_acf_ingredients( x
    , mean_acc, meansquare_acc, mean_delta_prod
    , start_lag = 1, stop_lag = 2
    , dim = (0,1) ):
  x_mean_acc += tf.math.reduce_mean( x, axis = dim )
  x_meansquare_acc += tf.math.reduce_mean( tf.math.square( x ), axis = dim  )
  for i, t in enumerate(range(start_lag, stop_lag)):
    deltat_prod = tf.math.multiply( x[:, t:], x[:, :-t]) if i > 0 else tf.math.square(x)
    mean_delta_prod[i] += tf.math.reduce_mean( deltat_prod, axis=dim )

