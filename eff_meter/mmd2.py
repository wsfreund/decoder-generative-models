# MMD functions implemented in tensorflow.
#   (adapted from https://github.com/ratschlab/RGAN/blob/master/mmd.py)

from .meter_base import ScalarEff, GenerativeEffMeter
from ..misc import *

import tensorflow as tf
import numpy as np

@tf.function
def sq_sum(t, name=None):
  "The squared Frobenius-type norm of a tensor, sum(t ** 2)."
  return tf.reduce_sum(tf.square(t))

@tf.function
def dot(x, y, name=None):
  "The dot product of two vectors x and y."
  x.get_shape().assert_has_rank(1)
  y.get_shape().assert_has_rank(1)
  return tf.squeeze(
      tf.matmul(
        tf.expand_dims(x, 0), 
        tf.expand_dims(y, 1)
      ))

class MMD2(ScalarEff,GenerativeEffMeter):
  """Quadratic-time MMD with Gaussian RBF kernel"""

  def __init__(self, name = "MMD2", sigmas = None, biased = True, **kw):
    super().__init__(name)
    self.sigmas                           = sigmas
    self.biased                           = biased
    self._n_sigmas                        = retrieve_kw( kw, "n_sigmas", 2 )
    self._biased_opt                      = retrieve_kw( kw, "biased_opt", True )
    #self._biased_opt                      = retrieve_kw( kw, "biased_opt", False )
    self._min_progress                    = retrieve_kw( kw, "beta_step_min_progress", 1e-3 )
    self._max_fails                       = retrieve_kw( kw, "max_fails", 500 )
    self._beta_opt_max_iter               = retrieve_kw( kw, "beta_opt_max_iter",      2000 )
    # TODO Optimization method is very brittle and prone to failure. Perharps
    # using golden ratio line search can help?
    self._beta_opt                        = retrieve_kw( kw, "beta_opt",  tf.keras.optimizers.RMSprop( learning_rate = 1e-2, clipvalue = 2. ) )
    self._beta_opt_initial_weight_values  = self._beta_opt.get_weights()

  @property
  def sigmas(self):
    return self._sigmas

  @sigmas.setter
  def sigmas(self, sigmas):
    if sigmas is not None:
      self._sigmas = tf.Variable( sigmas, dtype = tf.float32 )
      self._opt_betas = tf.Variable( tf.divide( 1., self.sigmas ), dtype = tf.float32 )
      self._betas = tf.Variable(self._opt_betas)
      self._wts = [1.0] * self.sigmas.get_shape()[0]
    else:
      self._sigmas = None

  def update_sigmas(self):
    self._betas.assign( self._opt_betas )
    self._sigmas.assign( tf.divide( 1., self._betas ) )

  def initialize(self, x_g1, x_g2, xmask_g1 = None, xmask_g2 = None):
    if self.initialized:
      return
    if xmask_g1 is not None or xmask_g2 is not None:
      raise NotImplementedError("MMD2 is not currently implemented for masked data")
    self.reset()
    if self.sigmas is None:
      import datetime
      start = datetime.datetime.now()
      print("Running %s optimization..." % self.__class__.__name__)
      # NOTE x_g1, x_g2 = 996 samples
      # NOTE code assumes dimensions are [batch_size, seq_length, num_generated_features]
      # FIXME In the origianl code, this power by -1 and 3 is quite strange and
      # is not refered anywhere in the paper. Also, it does not seem to be employed by
      # Sutherland bandwidth heuristic.
      # NOTE In the original code, training seems to be the only place where the
      # heuristic sigma is optimized. Anywhere else the heuristic sigma is
      # directly fed to the MMD computations.
      heuristic_sigma_training = median_pairwise_distance(x_g1, x_g2)
      self.sigmas = tf.Variable( initial_value = np.power(heuristic_sigma_training, np.linspace(-1., 3., num=self._n_sigmas))
                               , name='sigma', shape=self._n_sigmas, dtype=tf.float32 )
      # reset optimizer (i.e. clear first and second-order grad momentum history)
      self._beta_opt.set_weights(self._beta_opt_initial_weight_values)
      beta_iter = 0 
      best_t = prev_t = 0.
      fails = 0
      delta = 2*self._min_progress # set it to true
      prev_betas = tf.Variable( self._betas )
      # TODO Transform to tf.while
      while beta_iter < self._beta_opt_max_iter:
        #print("betas = %s" % self._betas)
        #print("best betas = %s" % self._opt_betas)
        prev_betas.assign( self._betas )
        mmd2, t_hat = self._opt_step( x_g1, x_g2 )
        if t_hat < best_t + self._min_progress:
          fails += 1 
        else:
          fails = 0
        if t_hat > best_t:
          self._opt_betas.assign( prev_betas )
          best_t = t_hat
          best_mmd2 = mmd2 
          #print("========== updating best beta: %s =======" % self._betas)
          #print("========== upading best t_hat: %s =======" % best_t)
        delta = t_hat - prev_t
        #print("t_hat = %f, t_hat_best = %f, delta = %f, fails = %d" % (t_hat, best_t, delta, fails) )
        prev_t = t_hat
        beta_iter += 1
        if fails == self._max_fails:
          break
      self.update_sigmas()
      # NOTE: we might consider caching k_xx
      #self.k_xx = self._compute_k_xx(x_data)
      total_time = datetime.datetime.now() - start
      print("Initialized %s in %s. mmd2 = %f. t_hat = %f" % (self.__class__.__name__, total_time, best_mmd2, best_t))
    self._x = x_g1
    self.initialized = True

  @tf.function
  def _opt_step( self, x_g1, x_g2 ):
    # NOTE: Do we want to keep track of the mmd2 variable for debugging purposes?
    with tf.GradientTape() as beta_tape:
      beta_tape.watch(self._betas)
      k_g1, k_g12, k_g2 = self._compute_k(x_g1,x_g2)
      mmd2, t_hat = _mmd2_and_ratio(k_g1, k_g12, k_g2, const_diagonal=tf.reduce_sum(self._wts), biased=self._biased_opt)
      # Following steepest ascent direction (i.e. maximizing t_hat)
      loss = tf.negative(t_hat)
    beta_grads = beta_tape.gradient(loss, self._betas)
    self._beta_opt.apply_gradients(zip([beta_grads], [self._betas]))
    return mmd2, t_hat 


  def accumulate(self, x_gen, x_mask = None):
    if x_mask is not None:
      raise NotImplementedError("MMD2 is not currently implemented for masked data")
    if self.i > 0:
      raise NotImplementedError("MMD2 is not able to work with multiple minibatches")
    self.start
    #import datetime
    #start = datetime.datetime.now()
    #print("Measuring %s efficiency..." % self.__class__.__name__)
    # FIXME: An open point that I am not completely sure is the heuristic and power setting of sigma.
    # Perharps it is good to take a look on the paper and original implementation. Also, double check
    # if this np.power linspace is also used in other parts of the code.
    # FIXME This reduce_sum wts seems to be an approximation which seems not to
    # be used in other references.
    k_xx, k_xy, k_yy = self._compute_k( self._x, x_gen )
    self.mmd2 = _mmd2( k_xx, k_xy, k_yy
                     , const_diagonal=tf.reduce_sum(self._wts)
                     , biased=self.biased )
    self.i += 1
    #total_time = datetime.datetime.now() - start
    #print("Measured %s efficiency in %s." % (self.__class__.__name__, total_time))
    self.stop
    return self.mmd2

  def retrieve(self):
    self.print
    return self.mmd2 

  def reset(self):
    super().reset()
    self.mmd2 = 0.
    self.i = 0

  @tf.function
  def _compute_k(self, X, Y):
    if len(X.shape) == 2:
      # matrix
      XX = tf.matmul(X, X, transpose_b=True)
      XY = tf.matmul(X, Y, transpose_b=True)
      YY = tf.matmul(Y, Y, transpose_b=True)
    elif len(X.shape) == 3:
      # tensor -- this is computing the Frobenius norm
      XX = tf.tensordot(X, X, axes=[[1, 2], [1, 2]])
      XY = tf.tensordot(X, Y, axes=[[1, 2], [1, 2]])
      YY = tf.tensordot(Y, Y, axes=[[1, 2], [1, 2]])
    # TODO X_sqnorms and K_XX can be cached
    X_sqnorms = tf.linalg.diag_part(XX)
    Y_sqnorms = tf.linalg.diag_part(YY)
    rX = tf.expand_dims(X_sqnorms,0)
    cX = tf.expand_dims(X_sqnorms,1)
    rY = tf.expand_dims(Y_sqnorms,0)
    cY = tf.expand_dims(Y_sqnorms,1)
    K_XX = tf.zeros_like(XX)
    K_XY = tf.zeros_like(XY)
    K_YY = tf.zeros_like(YY)
    for beta, wt in zip(tf.unstack(self._betas, axis=0), self._wts):
      gamma = tf.multiply( .5, tf.math.pow(beta,2.) )
      K_XX += rbf(wt,gamma,XX,cX,rX)
      K_XY += rbf(wt,gamma,XY,cX,rY)
      K_YY += rbf(wt,gamma,YY,cY,rY)
    return K_XX, K_XY, K_YY

@tf.function
def rbf(wt, gamma, mat, c, r ):
  return tf.multiply( wt
      , tf.exp( tf.negative( tf.multiply( gamma
                                        , tf.add(tf.multiply(-2.,mat), tf.add(c,r) )
                                        ))))


################################################################################
### Helper functions to compute variances based on kernel matrices
@tf.function
def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = tf.cast(K_XX.get_shape()[0], tf.float32)
    n = tf.cast(K_YY.get_shape()[0], tf.float32)

    if biased:
        mmd2 = (tf.reduce_sum(K_XX) / (m * m)
              + tf.reduce_sum(K_YY) / (n * n)
              - 2 * tf.reduce_sum(K_XY) / (m * n))
    else:
        if const_diagonal is not False:
            trace_X = m * const_diagonal
            trace_Y = n * const_diagonal
        else:
            trace_X = tf.trace(K_XX)
            trace_Y = tf.trace(K_YY)

        mmd2 = ((tf.reduce_sum(K_XX) - trace_X) / (m * (m - 1))
              + (tf.reduce_sum(K_YY) - trace_Y) / (n * (n - 1))
              - 2 * tf.reduce_sum(K_XY) / (m * n))

    return mmd2

_eps=1e-8
@tf.function
def _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=False,
                    min_var_est=_eps):
    mmd2, var_est = _mmd2_and_variance(
        K_XX, K_XY, K_YY, const_diagonal=const_diagonal, biased=biased)
    ratio = tf.divide( mmd2 , tf.sqrt(tf.maximum(var_est, min_var_est)) )
    return mmd2, ratio

@tf.function
def _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = tf.cast(K_XX.get_shape()[0], tf.float32)  # Assumes X, Y are same shape

    ### Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        const_diagonal = tf.cast(const_diagonal, tf.float32)
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
        sum_diag2_X = sum_diag2_Y = m * const_diagonal**2
    else:
        diag_X = tf.diag_part(K_XX)
        diag_Y = tf.diag_part(K_YY)

        sum_diag_X = tf.reduce_sum(diag_X)
        sum_diag_Y = tf.reduce_sum(diag_Y)

        sum_diag2_X = sq_sum(diag_X)
        sum_diag2_Y = sq_sum(diag_Y)

    Kt_XX_sums = tf.reduce_sum(K_XX, 1) - diag_X
    Kt_YY_sums = tf.reduce_sum(K_YY, 1) - diag_Y
    K_XY_sums_0 = tf.reduce_sum(K_XY, 0)
    K_XY_sums_1 = tf.reduce_sum(K_XY, 1)

    Kt_XX_sum = tf.reduce_sum(Kt_XX_sums)
    Kt_YY_sum = tf.reduce_sum(Kt_YY_sums)
    K_XY_sum = tf.reduce_sum(K_XY_sums_0)

    Kt_XX_2_sum = sq_sum(K_XX) - sum_diag2_X
    Kt_YY_2_sum = sq_sum(K_YY) - sum_diag2_Y
    K_XY_2_sum  = sq_sum(K_XY)

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
              + (Kt_YY_sum + sum_diag_Y) / (m * m)
              - 2 * K_XY_sum / (m * m))
    else:
        # NOTE Original code had a bug in this line
        mmd2 = (Kt_XX_sum / (m * (m-1))
              + Kt_YY_sum / (m * (m-1))
              - 2 * K_XY_sum / (m * m))

    var_est = (
          2 / (m**2 * (m-1)**2) * (
              2 * sq_sum(Kt_XX_sums) - Kt_XX_2_sum
            + 2 * sq_sum(Kt_YY_sums) - Kt_YY_2_sum)
        - (4*m-6) / (m**3 * (m-1)**3) * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 4*(m-2) / (m**3 * (m-1)**2) * (
              sq_sum(K_XY_sums_1) + sq_sum(K_XY_sums_0))
        - 4 * (m-3) / (m**3 * (m-1)**2) * K_XY_2_sum
        - (8*m - 12) / (m**5 * (m-1)) * K_XY_sum**2
        + 8 / (m**3 * (m-1)) * (
              1/m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - dot(Kt_XX_sums, K_XY_sums_1)
            - dot(Kt_YY_sums, K_XY_sums_0))
    )

    return mmd2, var_est


### additions from stephanie, for convenience
# TODO Make it become tensorflow oriented
def median_pairwise_distance(X, Y=None):
    """
    Heuristic for bandwidth of the RBF. Median pairwise distance of joint data.
    If Y is missing, just calculate it from X:
        this is so that, during training, as Y changes, we can use a fixed
        bandwidth (and save recalculating this each time we evaluated the mmd)
    At the end of training, we do the heuristic "correctly" by including
    both X and Y.

    Note: most of this code is assuming tensorflow, but X and Y are just ndarrays
    """
    if Y is None:
        Y = X       # this is horrendously inefficient, sorry
    if hasattr(X,'numpy'):
      X = X.numpy()
    if hasattr(Y,'numpy'):
      Y = Y.numpy()
   
    if len(X.shape) == 2:
        # matrix
        X_sqnorms = np.einsum('...i,...i', X, X)
        Y_sqnorms = np.einsum('...i,...i', Y, Y)
        XY = einsum('ia,ja', X, Y)
    elif len(X.shape) == 3:
        # tensor -- this is computing the Frobenius norm
        X_sqnorms = np.einsum('...ij,...ij', X, X)
        Y_sqnorms = np.einsum('...ij,...ij', Y, Y)
        XY = np.einsum('iab,jab', X, Y)
    else:
        raise ValueError(X)
    distances = np.sqrt(X_sqnorms.reshape(-1, 1) - 2*XY + Y_sqnorms.reshape(1, -1))
    return np.median(distances)
