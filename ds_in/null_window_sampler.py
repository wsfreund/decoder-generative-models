
from abc import ABC, abstractmethod
from ..misc import *

import numpy as np

class NullWindowSampler(object):

  def __init__( self, data_sampler, transformations
              , ds_for_sampling = "train"
              , ds_for_sampling_other = "val"
              , ensure_valid = lambda x: x):
    assert data_sampler is not None
    self.data_sampler = data_sampler
    self.transformations = transformations
    self.ds_for_sampling = ds_for_sampling
    self.ds_for_sampling_other = ds_for_sampling_other
    for t in self.transformations:
      t.use_sampler_as_default( data_sampler, ds_for_sampling_other )
      t.ensure_valid = ensure_valid

  def transform(self, sample, other = None ):
    null_data_list = []
    for transformation in self.transformations:
      null_data_list.append(transformation(sample, other))
    return null_data_list

  def sample(self):
    sample = data_sampler.sample(ds = self.ds_for_sampling)
    return self.transform(sample, ds)

  def plot_using_common_sample(self, sample = None, other = None, legend = None):
    from cycler import cycler
    import matplotlib.pyplot as plt
    import seaborn as sns
    if sample is None:
      sample = self.data_sampler.sample(ds = self.ds_for_sampling)
    plt.plot( np.squeeze( sample ), label = "Main sample" )
    if other is not None:
      plt.plot( np.squeeze( other ), label = "Other" )
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    cycle_colors  = cycler(color=sns.light_palette(colors[0], reverse = True, n_colors=len(self.transformations)+1))
    null_data_list = self.transform(sample, other)
    for i, (null_sample, cycle_info) in enumerate(zip(null_data_list, cycle_colors[1:])):
      label = "Null sample" + ( ( " (" + legend[i] + ")" ) if legend is not None else "")
      plt.plot( np.squeeze( null_sample ), '--', linewidth = .2, label = label, color = cycle_info['color'])
    plt.legend( prop={'size': 6} )

class NullWindowTransformation(ABC):

  def __init__( self, ensure_valid = lambda x: x
              , other_sampling_fcn = None):
    self.ensure_valid = ensure_valid
    self.other_sampling_fcn = other_sampling_fcn

  @abstractmethod
  def __call__(self, sample, other = None):
    return self.ensure_valid(sample)

  def __str__(self):
    return self.__class__.__name__

  def use_sampler_as_default(self, data_sampler, ds):
    if self.other_sampling_fcn is None:
      sampler_fcn = lambda s: self.data_sampler.sample( ds = ds )

class SamplePlusOther(NullWindowTransformation):

  def __init__(self, **kw):
    super().__init__(**kw)

  def __call__(self, sample, other = None):
    if other is None: 
      other = self.other_sampling_fcn(sample.shape)
    return super().__call__(sample + other)

class SamplePlusWhiteNoise(SamplePlusOther):
  def __init__(self, mu = 0., sigma = 1., **kw):
    super().__init__( other_sampling_fcn = lambda s: np.random.normal( loc = mu, scale = sigma, size = s ) )

class SamplePlusUniform(SamplePlusOther):
  def __init__(self, mu = 0., beta = 1., **kw):
    super().__init__( other_sampling_fcn = lambda s: np.random.uniform( mu-beta, mu+beta, size = s ) )

class SamplePlusCorrelatedNoise(SamplePlusOther):
  def __init__(self, mean, cov, **kw):
    super().__init__( other_sampling_fcn = lambda s: np.random.multivariate_normal( mean, cov, size = s[:-1] ) )

class InterpolateSamples(NullWindowTransformation):
  def __init__(self, alpha, **kw):
    super().__init__(**kw)
    assert 0. < alpha < 1. 
    self.alpha = alpha

  def __call__(self, sample, other = None):
    if other is None: 
      other = self.other_sampling_fcn(sample.shape)
    sample = (1-self.alpha)*sample + self.alpha*other
    return super().__call__(sample)

class InterpolateWithData(InterpolateSamples):
  def __init__(self, alpha, **kw):
    super().__init__(alpha, other_sampling_fcn = None, **kw)

class InterpolateWithWhiteNoise(InterpolateSamples):
  def __init__(self, alpha, mu = .5, sigma = 1., **kw):
    super().__init__(alpha, other_sampling_fcn = lambda s: np.random.normal( loc = mu, scale = sigma, size = s ), **kw)

class InterpolateWithUniform(InterpolateSamples):
  def __init__(self, alpha, mu, beta, **kw):
    super().__init__(alpha, other_sampling_fcn = lambda s: np.random.uniform( low = mu-beta, high = mu+beta, size = s ), **kw)

class InterpolateWithCorrelatedNoise(InterpolateSamples):
  def __init__(self, alpha, mean, cov, **kw):
    super().__init__( alpha, other_sampling_fcn = lambda s: np.random.multivariate_normal( mean, cov, size = s[:-1] ) )

class SwapTime(NullWindowTransformation):
  def __init__(self, max_swap, **kw):
    assert max_swap > 0
    super().__init__( **kw)
    self.max_swap = max_swap

  def __call__(self, sample, other = None):
    if hasattr( sample, "numpy" ):
      sample = sample.numpy()
    while True:
      orig = np.random.randint(0, sample.shape[-2], size = self.max_swap)
      dest = np.random.randint(0, sample.shape[-2], size = self.max_swap)
      if ( orig != dest ).all():
        break
    sample = sample.copy()
    copy = sample.copy()
    sample[...,dest,:] = sample[...,orig,:]
    sample[...,orig,:] = copy[...,dest,:]
    return super().__call__(sample)

class SwapFeature(NullWindowTransformation):
  def __init__(self, max_swap, **kw):
    assert max_swap > 0
    super().__init__( **kw)
    self.max_swap = max_swap

  def __call__(self, sample, other = None):
    if hasattr( sample, "numpy" ):
      sample = sample.numpy()
    while True:
      orig = np.random.randint(0, sample.shape[-1], size = self.max_swap)
      dest = np.random.randint(0, sample.shape[-1], size = self.max_swap)
      if ( orig != dest ).all():
        break
    sample = sample.copy()
    copy = sample.copy()
    sample[...,dest] = sample[...,orig]
    sample[...,orig] = copy[...,dest]
    return super().__call__(sample)

# FIXME SwapFeatureTime may not be working as expected
class SwapFeatureTime(NullWindowTransformation):
  def __init__(self, max_swap, **kw):
    assert max_swap > 0
    super().__init__( **kw)
    self.max_swap = max_swap

  def __call__(self, sample, other = None):
    if hasattr( sample, "numpy" ):
      sample = sample.numpy()
    while True:
      orig1 = np.random.randint(0, sample.shape[-2], size = self.max_swap)
      dest1 = np.random.randint(0, sample.shape[-2], size = self.max_swap)
      if ( orig1 != dest1 ).all():
        break
    while True:
      orig2 = np.random.randint(0, sample.shape[-1], size = self.max_swap)
      dest2 = np.random.randint(0, sample.shape[-1], size = self.max_swap)
      if ( orig2 != dest2 ).all():
        break
    sample = sample.copy()
    copy = sample.copy()
    sample[...,dest1,dest2] = sample[...,orig1,orig2]
    sample[...,orig1,orig2] = copy[...,dest1,dest2]
    return super().__call__(sample)

class RandomCrossOver(NullWindowTransformation):
  def __init__(self, p_swap, **kw):
    super().__init__(**kw)
    self.p_swap = p_swap

  def __call__(self, sample, other = None):
    if hasattr( sample, "numpy" ):
      sample = sample.numpy()
    sample = sample.copy()
    if other is None:
      other = self.other_sampling_fcn(sample.shape)
    if hasattr( other, "numpy" ):
      other = other.numpy()
    swap_mask = np.random.uniform(size=sample.shape) < self.p_swap 
    return super().__call__(self._swap(sample, other, swap_mask))

  def _swap(self, sample, other, swap_mask):
    sample[swap_mask] = other[swap_mask]
    return sample

class RandomCrossOverWithData(RandomCrossOver):
  def __init__(self, p_swap, **kw):
    super().__init__(p_swap, other_sampling_fcn = None)

class RandomCrossOverWithWhiteNoise(RandomCrossOver):
  def __init__(self, p_swap, mu = 0.5, sigma = 1., **kw):
    super().__init__(p_swap, other_sampling_fcn = lambda s: np.random.normal( loc = mu, scale = sigma, size = s), **kw )

class RandomCrossOverWithUniform(RandomCrossOver):
  def __init__(self, p_swap, mu = 0.5, beta = 1., **kw):
    super().__init__(p_swap, other_sampling_fcn = lambda s: np.random.uniform(  low = mu-beta, high = mu+beta, size = s), **kw )

class RandomCrossOverWithCorrelatedNoise(RandomCrossOver):
  def __init__(self, p_swap, mean, cov, **kw):
    super().__init__(p_swap, other_sampling_fcn = lambda s: np.random.multivariate_normal( mean, cov, size = s[:-1] ) )

class RandomCrossShuffle(RandomCrossOver):
  def __init__(self, p_swap, **kw):
    super().__init__(p_swap,**kw)
    self.rng = np.random.default_rng()

  def _swap(self, sample, other, swap_mask):
    dest = other[swap_mask]
    self.rng.shuffle(dest)
    sample[swap_mask] = dest
    return sample

class RandomCrossShuffleWithData(RandomCrossShuffle):
  def __init__(self, p_swap, **kw):
    super().__init__(p_swap, other_sampling_fcn = None)

class RandomCrossShuffleWithCorrelatedNoise(RandomCrossShuffle):
  def __init__(self, p_swap, mean, cov, **kw):
    super().__init__(p_swap, other_sampling_fcn = lambda s: np.random.multivariate_normal( mean, cov, size = s[:-1] ) )

# TODO
# - Mixing different examples using a random range or spliting windows
# - Add noise to FFT -> remove/change a fraction of frequencies (group in high/medium/low?)
#   - Note: This one is a bit more difficult because it requires long samples and subsampling within this longer sample
# - Apply moving averages and other smoothing methods/filters 
# - Other can be a function plus noise:
#   - Function can be a polynomial;
#   - Periodic
