import numpy as np
import tensorflow as tf
import sklearn
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import copy

try:
  from misc import *
  from sampler_base import SamplerBase
  from data_manager import TimeseriesMetadata
except ImportError:
  from .misc import *
  from .sampler_base import SamplerBase
  from .data_manager import TimeseriesMetadata

dev = False

class _CacheStorage(CleareableCache):
  cached_functions = []

class WindowSampler(SamplerBase):

  def __init__(self
      , cycle_width, cycle_shift
      , past_widths, overlaps
      , manager, **kw ):
    """
    TODO Help
    :param manager: a data manager instance 

    If running out of memory, increase sequence stride or reduce the
    shuffle_buffer_size_window
    """
    SamplerBase.__init__(self, manager, **kw)
    features                   = retrieve_kw(kw, "features",                   slice(None)                          )
    sample_from_marginals      = retrieve_kw(kw, "sample_from_marginals",      False                                )
    keep_marginal_axis         = retrieve_kw(kw, "keep_marginal_axis",         True                                 )
    past_widths_wrt_present    = retrieve_kw(kw, "past_widths_wrt_present",    True                                 )
    shuffle_buffer_size_window = retrieve_kw(kw, "shuffle_buffer_size_window", None                                 )
    sequence_stride            = retrieve_kw(kw, "sequence_stride",            1                                    )
    n_cycles                   = retrieve_kw(kw, "n_cycles",                   1                                    )
    del kw

    # Strip information from dataset
    self._date_time = manager.date_time
    self._time_step = manager.time_step

    # Work out the features
    self._features = features if features != slice(None) else manager.df.columns.to_numpy()
    #self.labels = labels if labels != slice(None) else manager.df.columns.to_numpy()
    self._feature_column_map = {l: i for i, l in enumerate(self._features)}
    #self.label_column_map = {l: i for i, l in enumerate(self.labels)}
    self._column_map = {name: i for i, name in enumerate(manager.df.columns)}

    # Work out the window parameters
    self._past_widths_wrt_present = past_widths_wrt_present 
    if self._past_widths_wrt_present:
      self._past_window_size = max(past_widths)
      self._past_widths = np.array(past_widths) - np.array([0] + past_widths[:-1])
      self._past_widths_up_to_mark_zero = past_widths
    else:
      self._past_window_size = sum(past_widths)
      self._past_widths = past_widths
      self._past_widths_up_to_mark_zero = list(np.cumsum(past_widths, dtype=np.int64))
    #self.label_width = label_width
    self._cycle_shift = cycle_shift
    self._cycle_width = cycle_width
    self._overlaps = overlaps

    #self.label_start = self.total_window_size - self.label_width
    #self.labels_slice = slice(self.label_start, None)
    #self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    # Compute the window width for n_cycles
    self.update_n_cycles(n_cycles)

    self._sequence_stride = sequence_stride
    self._sample_from_marginals = sample_from_marginals
    self._keep_marginal_axis = keep_marginal_axis
    if self._sample_from_marginals and shuffle_buffer_size_window is None:
      shuffle_buffer_size_window = TimeseriesMetadata.days_per_year #Metadata.days_per_month*3
    else:
      shuffle_buffer_size_window = TimeseriesMetadata.days_per_year
    if dev:
      shuffle_buffer_size_window = TimeseriesMetadata.days_per_week
    self._shuffle_buffer_size_window = shuffle_buffer_size_window
    return

  def clear_cache(self):
    _CacheStorage.clear_cached_functions()
    SamplingBase.clear_cache(self)

  def update_n_cycles(self, n_cycles):
    # Remove all cached information in order to recompute datasets
    _CacheStorage.clear_cached_functions()
    # Compute windows
    self.n_cycles = n_cycles
    self.present_mark_step = self._past_window_size+1
    self.future_window_size = n_cycles*self._cycle_shift
    self.total_window_size  = self._past_window_size + self.future_window_size

    # Compute slices
    self.full_past_slice = slice(0, self.total_window_size)
    self.full_window_indices = np.arange(self.total_window_size)[self.full_past_slice]
    if self._past_widths_wrt_present:
      edges = [0] + list(reversed(np.abs(np.array([0] + self._past_widths_up_to_mark_zero[:-1]) - np.ones_like(self._past_widths_up_to_mark_zero)*self._past_window_size)))
    else:
      edges = np.cumsum([0] + list(reversed(self._past_widths)))
    self.sliding_window_cycle_slice = slice(-self._cycle_width,None)
    self.all_past_slices = [slice(int(wstart), int(wend+g)) for wstart, wend, g in 
        # NOTE We reverse twice because it is easier to implement the algorithm
        # in reverse order, but it is better to keep the input slices on the
        # original order
        zip( reversed(edges[:-1])
           , reversed(edges[1:])
           , self._overlaps)]
    self.all_past_indices = [self.full_window_indices[s] for s in self.all_past_slices]


  def plot(self, plot_col = None, model = None,  ds = "val", n_examples = 3):
    from cycler import cycler
    plt.figure(figsize=(12, 8))
    # These allow to choose which farm to sample from
    if plot_col is None:
      if not self._sample_from_marginals:
        plot_col = self._features[0]
    if not self._sample_from_marginals:
      feature_plot_col_index = self._feature_column_map.get(plot_col,None)
      #label_plot_col_index = self.label_column_map.get(plot_col,None)
    else:
      feature_plot_col_index = slice(None)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ['+', 'x', 'X', '*', '.', 'o']
    cycle_markers = cycler(marker=markers)
    cycle_colors  = cycler(color=sns.light_palette(colors[0], reverse=True, n_colors=self.n_cycles))
    cycle_cycler  = cycle_markers * cycle_colors
    input_markers = cycler(marker=markers)
    input_colors  = cycler(color=colors[1:])
    input_cycler  = input_markers * input_colors
    #day_zero = datetime.datetime(year=2000,day=1,month=1)
    #delta = pd.DatetimeIndex(day_zero+(np.array(self.full_window_indices)-self.present_mark_step)*self._time_step)
    delta = (np.array(self.full_window_indices)-self.present_mark_step)
    for n in range(n_examples):
      plt.subplot(n_examples, 1, n+1)
      inputs_dict = self.sample( ds = ds )
      if feature_plot_col_index is not None or self._sample_from_marginals:
        slices = [d[:,:,feature_plot_col_index] for d in inputs_dict["slices"]]
        for i, (idxs, s, input_opts) in enumerate(zip(self.all_past_indices, slices, input_cycler)):
          plt.plot(delta[idxs], np.squeeze(self._pp.inverse_transform(s)[0,:,:])
              , label='Past period [%d]' % i, markersize=2
              , markeredgewidth=1, **input_opts, alpha = 0.7)
      for c, cycle_opts in zip(range(self.n_cycles), cycle_cycler):
        cycle = inputs_dict["cycle"][c,:,feature_plot_col_index]
        if plot_col: plt.ylabel(f'{plot_col}')
        if feature_plot_col_index is not None or self._sample_from_marginals:
          plt.plot(delta[self._cycle_indices(c)], np.squeeze(self._pp.inverse_transform(cycle))
              , label=('Cycle [%d]' % c) if c in (0,self.n_cycles-1) else '', markersize=2
              , markeredgewidth=1, alpha = 0.7
              , **cycle_opts)
        if model is not None:
          # TODO
          outputs = model(inputs)
          plt.scatter(self.input_indices, np.squeeze(outputs[:,:,feature_plot_col_index]) # label_indices, feature_plot_col_index
            , label='Output' , marker = 'x' )
                        #marker='X', edgecolors='k', label='Predictions',
                        #c='#ff7f0e', s=64)
        if n == 0:
          plt.legend(loc=2)
    plt.xlabel('Time [%s]' % self._date_time.freq)
    return

  def _cycle_slice(self, cycle_idx=0):
    end = -(self.n_cycles-(cycle_idx+1))*self._cycle_shift
    start = end - self._cycle_width
    return slice(start,end if end != 0 else None)

  def _cycle_indices(self, cycle_idx=0):
    return self.full_window_indices[self._cycle_slice(cycle_idx)]

  def _make_dataset(self, df, batch = True):
    # This will not work if attempting to forecast the past:
    try:
      sklearn.utils.validation.check_is_fitted( self._pp )
    except sklearn.exceptions.NotFittedError:
      self._pp.fit(self.train_df)
    n_examples, n_dims = df.shape
    # NOTE This slice on features must be removed when adding support to labels
    data = self._pp.transform( df.loc[:,self._features].to_numpy(dtype=np.float32) )
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=self._sequence_stride,
        shuffle=False,
        batch_size=tf.constant(1,dtype=tf.int64),)
    # XXX Hack to remove batching
    ds = ds._input_dataset
    if self._sample_from_marginals:
      ds = self._pattern_sampler(ds)
    if self._shuffle:
      shuffle_kwargs = copy.copy(self._shuffle_kwargs)
      if not "buffer_size" in shuffle_kwargs:
        # Default is set to shuffle a full year in the buffer
        val = int(np.round(datetime.timedelta(days=self._shuffle_buffer_size_window)/(self._sequence_stride*self._time_step)))
        if self._sample_from_marginals:
          val *= len(self._features)
        if val < self._batch_size * 8:
          val = self._batch_size * 8
        shuffle_kwargs["buffer_size"] = val
      ds = ds.shuffle(**shuffle_kwargs)
    ds = self._nested_timeseries(
        start_index = 0,
        end_index = self.total_window_size,
        input_dataset = ds,
        sequence_length=self._past_window_size+self._cycle_shift,
        sequence_stride=self._cycle_shift)
    ds = ds.map(self._extract_data)
    if batch:
      ds = ds.batch(self._batch_size, drop_remainder = True)
    # shuffle
    return ds

  def _extract_data(self, window):
    # Slice
    slices = []
    for s in self.all_past_slices:
      slices.append( window[:,s] if self._sample_from_marginals and not self._keep_marginal_axis else window[:,s,:] ) 
    cycle = window[:,self.sliding_window_cycle_slice] if self._sample_from_marginals and not self._keep_marginal_axis else window[:,self.sliding_window_cycle_slice,:]
    #if self._features is not None: # Apply slicing on window
    #  inputs = tf.stack(
    #      [inputs[:, self._column_map[name]] for name in self._features],
    #      axis=-1)
    ##inputs = tf.expand_dims( inputs, axis = 1 )
    #inputs.set_shape([self.n_cycles, self._past_widths, None])
    return { "cycle" : cycle, "slices" : tuple(slices) }

  def _pattern_sampler(self, input_dataset):
    from tensorflow.python.data.ops import dataset_ops
    from tensorflow.python.ops import array_ops
    from tensorflow.python.ops import math_ops
    # Determine the lowest dtype to store start positions (to lower memory usage).
    num_seqs = len(self._features)
    index_dtype = 'int32' if num_seqs < 2147483647 else 'int64'

    positions = np.arange(0, num_seqs, dtype=index_dtype)
    feature_selection_ds = dataset_ops.Dataset.from_tensor_slices(positions)
    def cycle_select(array):
      def select(steps, pat): 
        return tf.expand_dims( steps[:,pat], axis = -1 ) if self._keep_marginal_axis else steps[:,pat]
      return dataset_ops.Dataset.zip(
          (dataset_ops.Dataset.from_tensors(array).repeat(), feature_selection_ds)
        ).map(select)
    dataset = input_dataset.interleave(
        cycle_select
      , cycle_length = 1
      , block_length = 1
    )
    return dataset

  def _nested_timeseries(self,
      input_dataset,
      sequence_length,
      start_index,
      end_index,
      sequence_stride=1,
      sampling_rate=1):
    from tensorflow.python.data.ops import dataset_ops
    from tensorflow.python.ops import array_ops
    from tensorflow.python.ops import math_ops
    # Determine the lowest dtype to store start positions (to lower memory usage).
    num_seqs = end_index - start_index - (sequence_length * sampling_rate) + 1
    index_dtype = 'int32' if num_seqs < 2147483647 else 'int64'

    # Generate start positions
    start_positions = np.arange(0, num_seqs, sequence_stride, dtype=index_dtype)
    positions_ds = dataset_ops.Dataset.from_tensors(start_positions).repeat()

    sequence_length = math_ops.cast(sequence_length, dtype=index_dtype)
    sampling_rate = math_ops.cast(sampling_rate, dtype=index_dtype)

    def indices_fcn( i, positions ): 
      return math_ops.range( positions[i],
                      positions[i] + sequence_length * sampling_rate,
                      sampling_rate)

    # For each initial window position, generates indices of the window elements
    indices_ds = dataset_ops.Dataset.zip(
        (dataset_ops.Dataset.range(len(start_positions)), positions_ds)).map(
            indices_fcn
        )
    def cycle_data(array):
      def gather(steps, inds): 
        return array_ops.gather(steps, inds, axis = 0)
      return dataset_ops.Dataset.zip(
          (dataset_ops.Dataset.from_tensors(array[start_index:end_index,:] if not self._sample_from_marginals else array[start_index:end_index]).repeat(), indices_ds)
        ).map(gather).batch(self.n_cycles, drop_remainder = True)
    # Note that we are using a nested dataset structure
    dataset = input_dataset.interleave(
        cycle_data
      , cycle_length = 1
      , block_length = 1
    )
    return dataset

  def _split_data(self, df, val_frac, test_frac ):
    # TODO Add compatibility with time-series cross-validation
    # Note that we split without shuffling.
    # The goal is to have realistic generalization conditions, i.e. generalize
    # for future data
    # NOTE: If we are going to generalize for future and past data, this needs
    # to be taken into account.
    n = len(df)
    full_train_frac = 1. - test_frac
    train_frac = 1. - val_frac

    full_train_slice = slice(0,int(n*full_train_frac))
    full_train_df = df[full_train_slice]
    n_tr = int(n*full_train_frac)
    train_slice=slice(0,int(n_tr*full_train_frac))
    self.train_df = full_train_df[train_slice]

    self.val_slice=slice(int(n_tr*train_frac),None)
    self.val_df = full_train_df[self.val_slice]
    self.val_df.reset_index(drop=True,inplace=True)

    self.test_slice=slice(int(n*(1.-full_train_frac)),None)
    self.test_df = df[self.test_slice]
    self.test_df.reset_index(drop=True,inplace=True)
    return

  def __repr__(self):
    return '\n'.join([
        f'full window slice: {self.full_past_slice}',
        f'#cycles: {self.n_cycles}; cycle width: {self._cycle_width}; cycle shift: {self._cycle_shift}',
        f'past window widths: {self._past_widths}',
        f'past window widths w.r.t. present: {self._past_widths_up_to_mark_zero}',
        f'past window overlap: {self._overlaps}',
        f'feature column name(s): {self._features}',
        ])

