from .meter_base import GenerativeEffMeter, GenerativeEffBufferedMeter
from ..misc import *

import tensorflow as tf

class ACFMeter(GenerativeEffMeter):

  def __init__(self, name = "ACF", data_parser = lambda x: x, gen_parser = lambda x: x, start_lag = 1, stop_lag = 2, **kw):
    super().__init__(name = name, data_parser = data_parser, gen_parser = gen_parser, **kw)
    self.start_lag                   = start_lag
    self.stop_lag                    = stop_lag
    #
    self.xreal_mean_acc              = tf.Variable([0.], dtype = tf.float32)
    self.xreal_mean_square_acc       = tf.Variable([0.], dtype = tf.float32)
    self.xreal_mean_delta_t_prod_acc = None
    self.xreal_mean_t_pos_acc        = tf.Variable([0.], dtype = tf.float32)
    self.xreal_mean_t_neg_acc        = tf.Variable([0.], dtype = tf.float32)
    #
    self.xgen_mean_acc               = tf.Variable([0.], dtype = tf.float32)
    self.xgen_mean_square_acc        = tf.Variable([0.], dtype = tf.float32)
    self.xgen_mean_delta_t_prod_acc  = None
    self.xgen_mean_t_pos_acc         = tf.Variable([0.], dtype = tf.float32)
    self.xgen_mean_t_neg_acc         = tf.Variable([0.], dtype = tf.float32)

  def update_on_parsed_data(self, data, mask = None):
    if mask is not None:
      raise NotImplementedError("%s is not currently implemented for masked data" % self.__class__.__name__)
    if self.xreal_mean_delta_t_prod_acc is None:
      self.xreal_mean_delta_t_prod_acc = tf.Variable([[0.]*(self.stop_lag-self.start_lag)]*data.shape[-1], dtype = tf.float32)
    accumulate_acf_ingredients( data
                              , self.xreal_mean_acc, self.xreal_mean_square_acc
                              , self.xreal_mean_delta_t_prod_acc, self.xreal_mean_t_pos_acc, self.xreal_mean_t_neg_acc
                              , self.start_lag, self.stop_lag )

  def update_on_parsed_gen(self, data, mask = None):
    if mask is not None:
      raise NotImplementedError("%s is not currently implemented for masked data" % self.__class__.__name__)
    if self.xgen_mean_delta_t_prod_acc is None:
      self.xgen_mean_delta_t_prod_acc = tf.Variable([[0.]*(self.stop_lag-self.start_lag)]*data.shape[-1], dtype = tf.float32)
    accumulate_acf_ingredients( data
                              , self.xgen_mean_acc, self.xgen_mean_square_acc
                              , self.xgen_mean_delta_t_prod_acc, self.xgen_mean_t_pos_acc, self.xgen_mean_t_neg_acc
                              , self.start_lag, self.stop_lag )

  def retrieve(self):
    self.start
    if not self._locked_data_statistics:
      self.acf_data = self._compute_acf( self.xreal_mean_acc, self.xreal_mean_square_acc
                                       , self.xreal_mean_delta_t_prod_acc, self.xreal_mean_t_pos_acc, self.xreal_mean_t_neg_acc
                                       , self.data_batch_counter )
      self._locked_data_statistics = True
    self.acf_gen = self._compute_acf( self.xgen_mean_acc, self.xgen_mean_square_acc
                                    , self.xgen_mean_delta_t_prod_acc, self.xgen_mean_t_pos_acc, self.xgen_mean_t_neg_acc
                                    , self.gen_batch_counter )
    total_diff = self._compute_stats( self.acf_data, self.acf_gen )
    self.stop
    return { self.name + ( '_lag%d' % lag) : total_diff[i] for i, lag in enumerate(range(self.start_lag,self.stop_lag)) }

  @tf.function
  def _compute_acf( self, mean_acc, mean_square_acc
                  , mean_delta_t_prod_acc, mean_t_pos_acc, mean_t_neg_acc
                  , counter ):
    mean                  = tf.divide(mean_acc, counter)
    mean_square           = tf.divide(mean_square_acc, counter)
    mean_delta_t_prod     = tf.divide(mean_delta_t_prod_acc, counter)
    mean_t_pos            = tf.divide(mean_t_pos_acc, counter)
    mean_t_neg            = tf.divide(mean_t_neg_acc, counter)
    acf = tf.math.divide( tf.math.subtract( mean_delta_t_prod, tf.math.multiply(mean_t_pos, mean_t_neg) )
                        , tf.math.subtract( mean_square, tf.square(mean) ) )
    return acf

  @tf.function
  def _compute_stats(self, acf_data, acf_gen ):
    # NOTE Norm-2 is employed (as in the original code) although norm-1 is
    # referred on the paper
    return tf.linalg.norm( tf.math.subtract( self.acf_gen, self.acf_data)
                         , ord = 2, axis = 0 )

  def reset(self):
    super().reset()
    def reset(t): t.assign(tf.zeros_like(t))
    reset(self.xgen_mean_acc)
    reset(self.xgen_mean_square_acc)
    reset(self.xgen_mean_delta_t_prod_acc)
    reset(self.xgen_mean_t_pos_acc)
    reset(self.xgen_mean_t_neg_acc)

class ACFBufferedMeter(GenerativeEffBufferedMeter,ACFMeter):

  def __init__( self, name = "ACF", data_parser = lambda x: x, gen_parser = lambda x: x, start_lag = 1, stop_lag = 2
              , data_buffer = None, gen_buffer = None, max_buffer_size = 16, **kw):
    super().__init__( name = name, data_parser = data_parser, gen_parser = gen_parser, start_lag = start_lag, stop_lag = stop_lag
                    , data_buffer = data_buffer, gen_buffer = gen_buffer, max_buffer_size = max_buffer_size, **kw)

@tf.function
def accumulate_acf_ingredients( x
    , mean_acc, mean_square_acc
    , mean_delta_prod_acc, mean_t_pos_acc, mean_t_neg_acc
    , start_lag = 1, stop_lag = 2
    , dim = (0,1) ):
  mean_acc.assign_add( tf.math.reduce_mean( x, axis = dim ) )
  mean_square_acc.assign_add( tf.math.reduce_mean( tf.math.square( x ), axis = dim  ) )
  #
  l_mean_delta_prod = []
  l_mean_t_pos = []
  l_mean_t_neg = []
  #
  for t in range(start_lag, stop_lag):
    deltat_prod = tf.math.multiply( x[:, t:], x[:, :-t]) if t > 0 else tf.math.square(x)
    # TODO Try assign add to slice here
    l_mean_delta_prod.append( tf.math.reduce_mean( deltat_prod, axis=dim ) )
    l_mean_t_pos.append( tf.math.reduce_mean( x[:, t:] ) )
    l_mean_t_neg.append( tf.math.reduce_mean( x[:, :-t] ) )
  #
  l_mean_delta_prod = tf.stack(l_mean_delta_prod)
  l_mean_t_pos = tf.stack(l_mean_t_pos)
  l_mean_t_neg = tf.stack(l_mean_t_neg)
  #
  mean_delta_prod_acc.assign_add( l_mean_delta_prod )
  mean_t_pos_acc.assign_add( l_mean_t_pos )
  mean_t_neg_acc.assign_add( l_mean_t_neg )

# For debugging purposes only
@tf.function
def _acf_tf_single_batch(x, start_lag = 1, stop_lag = 2, dim=(0,1)):
  acf_list = list()
  #x = x - tf.math.reduce_mean(x, axis = (0,1) )
  var = tf.math.reduce_variance(x, axis = (0,1) )
  for i in range(start_lag, stop_lag):
    if i > 0:
      x_pos = x[:, i:]
      x_neg = x[:, :-i]
      y = tf.math.multiply( tf.math.subtract(x_pos, tf.math.reduce_mean(x_pos,axis=(0,1)))
                          , tf.math.subtract(x_neg, tf.math.reduce_mean(x_neg,axis=(0,1))))
    else:
      y = tf.square(tf.subtract(x,tf.math.reduce_mean(x,axis=(0,1))))
    acf_i = tf.math.reduce_mean(y, axis=dim ) / var
    acf_list.append(acf_i)
  if dim==(0,1):
    return tf.stack(acf_list)
  else:
    return tf.concat(acf_list, axis = 1)
