from .meter_base import GenerativeEffMeter, GenerativeEffBufferedMeter
from ..misc import *

import tensorflow as tf

class ACFMeter(GenerativeEffMeter):

  def __init__(self, n_features, t_dim_len, name = "ACF", data_parser = lambda x: x, gen_parser = lambda x: x, start_lag = 1, stop_lag = 2, **kw):
    super().__init__(name = name, data_parser = data_parser, gen_parser = gen_parser, **kw)
    self.start_lag                   = start_lag
    self.stop_lag                    = stop_lag
    #
    self.t_dim_len                   = tf.constant( t_dim_len, dtype = tf.float32 )
    self.t_lag_span                  = tf.subtract( self.t_dim_len, tf.range(start_lag, stop_lag, dtype = tf.float32 ) )
    #
    self._allocate_variables( 'main', (n_features,), (stop_lag-start_lag, n_features) )
    # FIXME
    self.data_list = []
    self.gen_list = []

  def _allocate_variables( self, label, shape, t_shape ):
    def allocate():
      cont = Container()
      cont.data_acc              = tf.Variable(tf.zeros(shape), dtype = tf.float32)
      cont.data_square_acc       = tf.Variable(tf.zeros(shape), dtype = tf.float32)
      cont.data_delta_t_prod_acc = tf.Variable(tf.zeros(t_shape), dtype = tf.float32)
      cont.data_t_pos_acc        = tf.Variable(tf.zeros(t_shape), dtype = tf.float32)
      cont.data_t_neg_acc        = tf.Variable(tf.zeros(t_shape), dtype = tf.float32)
      cont.counter               = tf.Variable(0, dtype = tf.int64)
      return cont
    xdata_cont = allocate()
    xgen_cont = allocate()
    label_cont = Container()
    label_cont.xdata = xdata_cont
    label_cont.xgen = xgen_cont
    setattr(self, label, label_cont)

  def update_on_parsed_data(self, data, mask = None, corr_new = 1., corr_tot = 1.):
    if mask is not None:
      raise NotImplementedError("%s is not currently implemented for masked data" % self.__class__.__name__)
    cont = self.main.xdata
    accumulate_acf_ingredients( data
                              , cont.data_acc, cont.data_square_acc
                              , cont.data_delta_t_prod_acc, cont.data_t_pos_acc, cont.data_t_neg_acc
                              , cont.counter
                              , self.start_lag, self.stop_lag )

  def update_on_parsed_gen(self, data, mask = None, corr_new = 1., corr_tot = 1.):
    if mask is not None:
      raise NotImplementedError("%s is not currently implemented for masked data" % self.__class__.__name__)
    cont = self.main.xgen
    accumulate_acf_ingredients( data 
                              , cont.data_acc, cont.data_square_acc
                              , cont.data_delta_t_prod_acc, cont.data_t_pos_acc, cont.data_t_neg_acc
                              , cont.counter
                              , self.start_lag, self.stop_lag )


  def retrieve(self):
    self.start
    if not self._locked_data_statistics:
      root_cont = self.main
      cont = root_cont.xdata
      root_cont.acf_data = self._compute_acf( cont.data_acc, cont.data_square_acc
                                            , cont.data_delta_t_prod_acc, cont.data_t_pos_acc, cont.data_t_neg_acc
                                            , cont.counter )
      self._locked_data_statistics = True
    # 
    cont = root_cont.xgen
    root_cont.acf_gen = self._compute_acf( cont.data_acc, cont.data_square_acc
                                         , cont.data_delta_t_prod_acc, cont.data_t_pos_acc, cont.data_t_neg_acc
                                         , cont.counter )
    root_cont.total_diff = self._compute_stats( root_cont.acf_data, root_cont.acf_gen )
    return { self.name : root_cont.total_diff }

  @tf.function
  def _compute_acf( self, data_acc, data_square_acc
                  , data_delta_t_prod_acc, data_t_pos_acc, data_t_neg_acc
                  , counter ):
    counter               = tf.cast(counter, tf.float32)
    mean                  = tf.divide(data_acc,              tf.multiply(counter, self.t_dim_len))
    mean_square           = tf.divide(data_square_acc,       tf.multiply(counter, self.t_dim_len))
    mean_delta_t_prod     = tf.divide(data_delta_t_prod_acc, tf.multiply(counter, self.t_lag_span))
    mean_t_pos            = tf.divide(data_t_pos_acc,        tf.multiply(counter, self.t_lag_span))
    mean_t_neg            = tf.divide(data_t_neg_acc,        tf.multiply(counter, self.t_lag_span))
    acf = tf.math.divide( tf.math.subtract( mean_delta_t_prod, tf.math.multiply(mean_t_pos, mean_t_neg) ) , tf.math.subtract( mean_square, tf.square(mean) ) )
    return acf

  @tf.function
  def _compute_stats(self, acf_data, acf_gen ):
    # NOTE Norm-2 is employed (as in the original code) although norm-1 is
    # referred on the paper
    return tf.linalg.norm( tf.math.subtract( acf_gen, acf_data), ord = 2, axis = 0 )

  def reset(self):
    super().reset()
    def reset(t): t.assign(tf.zeros_like(t))
    cont = self.main.xgen
    reset(cont.data_acc)
    reset(cont.data_square_acc)
    reset(cont.data_delta_t_prod_acc)
    reset(cont.data_t_pos_acc)
    reset(cont.data_t_neg_acc)

class ACFBufferedMeter(GenerativeEffBufferedMeter,ACFMeter):

  def __init__( self, n_features, t_dim_len, name = "ACF", data_parser = lambda x: x, gen_parser = lambda x: x, start_lag = 1, stop_lag = 2
              , data_buffer = None, gen_buffer = None, max_buffer_size = 16, **kw):
    super().__init__( n_features = n_features, t_dim_len = t_dim_len, name = name, data_parser = data_parser, gen_parser = gen_parser
                    , start_lag = start_lag, stop_lag = stop_lag
                    , data_buffer = data_buffer, gen_buffer = gen_buffer, max_buffer_size = max_buffer_size, **kw)

@tf.function
def accumulate_acf_ingredients( x
    , data_acc, data_square_acc
    , data_delta_prod_acc, data_t_pos_acc, data_t_neg_acc
    , data_counter
    , start_lag = 1, stop_lag = 2
    , dim = (0,1) ):
  data_acc.assign_add( tf.math.reduce_sum( x, axis = dim ) )
  data_square_acc.assign_add( tf.math.reduce_sum( tf.math.square( x ), axis = dim  ) )
  #
  l_data_delta_prod = []
  l_data_t_pos = []
  l_data_t_neg = []
  #
  for t in range(start_lag, stop_lag):
    deltat_prod = tf.math.multiply( x[:, t:], x[:, :-t]) if t > 0 else tf.math.square(x)
    # TODO Try assign add to slice here
    l_data_delta_prod.append( tf.math.reduce_sum( deltat_prod, axis=dim ) )
    l_data_t_pos.append( tf.math.reduce_sum( x[:, t:], axis=dim ) )
    l_data_t_neg.append( tf.math.reduce_sum( x[:, :-t], axis=dim ) )
  #
  l_data_delta_prod = tf.stack(l_data_delta_prod, axis = 0) # XXX Not tested
  l_data_t_pos = tf.stack(l_data_t_pos, axis = 0)
  l_data_t_neg = tf.stack(l_data_t_neg, axis = 0)
  #
  data_delta_prod_acc.assign_add( l_data_delta_prod )
  data_t_pos_acc.assign_add( l_data_t_pos )
  data_t_neg_acc.assign_add( l_data_t_neg )
  data_counter.assign_add( tf.shape(x, out_type=tf.int64 )[0] )

