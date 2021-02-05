import numpy as np
import tensorflow as tf
import sklearn
import datetime
import copy

from ..misc import *
from .sampler_base import SamplerBase, SpecificFlowSamplingOpts
from .data_manager import TimeseriesMetadata

class SpecificFlowWindowSamplingOpts(SpecificFlowSamplingOpts):
  """
  Allows to specify different sampling options for the same dataset
  """
  def set_unset_to_default(self, sampler, df):
    super().set_unset_to_default( sampler, df )
    if not "buffer_size" in self.shuffle_kwargs:
      if sampler.shuffle_buffer_size_window is NotSet:
        shuffle_buffer_size_window = datetime.timedelta(days=TimeseriesMetadata.days_per_year)
      else:
        assert isinstance(sampler.shuffle_buffer_size_window, datetime.timedelta)
        shuffle_buffer_size_window = sampler.shuffle_buffer_size_window
      # Compute the buffer size
      val = int(np.round( shuffle_buffer_size_window
                        / (sampler._sequence_stride*sampler._time_step)
               ))
      # Increase buffer size when sampling from marginals
      if sampler._sample_from_marginals:
        val *= len(sampler._features)
      if self.batch_size is not None and val < self.batch_size * 8:
        val = self.batch_size * 8
      self.shuffle_kwargs["buffer_size"] = val

class _CacheStorage(CleareableCache):
  cached_functions = []

class WindowSampler(SamplerBase):

  def __init__(self, manager, cycle_width, cycle_shift, **kw ):
    """
    TODO Help
    :param manager: a data manager instance 
    """
    if "specific_flow_sampling_opt_class" not in kw:
      kw["specific_flow_sampling_opt_class"] = SpecificFlowWindowSamplingOpts
    super().__init__(manager, **kw)
    self.shuffle_buffer_size_window = retrieve_kw(kw, "shuffle_buffer_size_window" )
    self._sample_from_marginals     = retrieve_kw(kw, "sample_from_marginals",      False                                )
    self._keep_marginal_axis        = retrieve_kw(kw, "keep_marginal_axis",         True                                 )
    self._sequence_stride           = retrieve_kw(kw, "sequence_stride",            1                                    )
    self._past_widths_wrt_present   = retrieve_kw(kw, "past_widths_wrt_present",    True                                 )
    self._overlaps                  = retrieve_kw(kw, "overlaps",                   None                                 )
    features                        = retrieve_kw(kw, "features",                   []                                   )
    past_widths                     = retrieve_kw(kw, "past_widths",                None                                 )
    n_cycles                        = retrieve_kw(kw, "n_cycles",                   1                                    )
    del kw

    # Strip information from dataset
    self._date_time = manager.date_time
    self._time_step = manager.time_step

    # Work out the features
    self._features = features if features else manager.df.columns.to_numpy()
    #self.labels = labels if labels != slice(None) else manager.df.columns.to_numpy()
    self._feature_column_map = {l: i for i, l in enumerate(self._features)}
    #self.label_column_map = {l: i for i, l in enumerate(self.labels)}
    self._column_map = {name: i for i, name in enumerate(manager.df.columns)}

    # Work out the window parameters
    if past_widths is None:
      self._past_widths = past_widths
      self._past_window_size = cycle_width - cycle_shift
      self._past_widths_up_to_mark_zero = []
    else:
      if self._overlaps is None:
        self._overlaps = [0]*len(past_widths)
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

    #self.label_start = self.total_window_size - self.label_width
    #self.labels_slice = slice(self.label_start, None)
    #self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    # Compute the window width for n_cycles
    self.update_n_cycles(n_cycles)
    return

  def clear_cache(self):
    _CacheStorage.clear_cached_functions()
    SamplingBase.clear_cache(self)

  def update_n_cycles(self, n_cycles):
    # Remove all cached information in order to recompute datasets
    _CacheStorage.clear_cached_functions()
    # Compute windows
    self.n_cycles = n_cycles
    self.present_mark_step = self._past_window_size if self._past_widths is not None else 0
    self.future_window_size = n_cycles*self._cycle_shift
    self.total_window_size  = self._past_window_size + self.future_window_size

    # Compute slices
    self.full_past_slice = slice(0, self.total_window_size)
    self.full_window_indices = np.arange(self.total_window_size)[self.full_past_slice]
    self.sliding_window_cycle_slice = slice(-self._cycle_width,None)
    if self._past_widths is not None:
      if self._past_widths_wrt_present:
        edges = [0] + list(reversed(np.abs(np.array([0] + self._past_widths_up_to_mark_zero[:-1]) - np.ones_like(self._past_widths_up_to_mark_zero)*self._past_window_size)))
      else:
        edges = np.cumsum([0] + list(reversed(self._past_widths)))
      self.all_past_slices = [slice(int(wstart), int(wend+g)) for wstart, wend, g in 
          # NOTE We reverse twice because it is easier to implement the algorithm
          # in reverse order, but it is better to keep the input slices on the
          # original order
          zip( reversed(edges[:-1])
             , reversed(edges[1:])
             , self._overlaps)]
      self.all_past_indices = [self.full_window_indices[s] for s in self.all_past_slices]
    else:
      self.all_past_slices = []
      self.all_past_indices = []


  def plot(self, plot_col = None, model = None,  ds = "val", n_examples = 3):
    from cycler import cycler
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(12, 8))
    # These allow to choose which farm to sample from
    if plot_col is None:
      if not self._sample_from_marginals:
        plot_col = self._features[0]
    if not self._sample_from_marginals:
      plot_col = self._feature_column_map.get(plot_col,None)
      feature_plot_col_index = slice(plot_col, plot_col+1) if plot_col is not None else slice(None)
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
      inputs = self.sample( ds = ds )
      # Plot slices
      if feature_plot_col_index is not None or self._sample_from_marginals:
        slices = [d[:,:,feature_plot_col_index] for d in inputs["slices"]] if isinstance(inputs, dict) else []
        for i, (idxs, s, input_opts) in enumerate(zip(self.all_past_indices, slices, input_cycler)):
          plt.plot(delta[idxs], np.squeeze(s[0,:,:])#self._pp.inverse_transform(s)[0,:,:])
              , label='Past period [%d]' % i, markersize=2
              , markeredgewidth=1, **input_opts, alpha = 0.7)
      # Plot cycles
      for c, cycle_opts in zip(range(self.n_cycles), cycle_cycler):
        cycle = inputs["cycle"][c,:,feature_plot_col_index] if isinstance(inputs, dict) else inputs
        if plot_col: plt.ylabel(f'{plot_col}')
        if feature_plot_col_index is not None or self._sample_from_marginals:
          plt.plot(delta[self._cycle_indices(c)], np.squeeze(cycle)#self._pp.inverse_transform(cycle))
              , label=(('Cycle [%d]' % c) if c in (0,self.n_cycles-1) else '') if self.n_cycles>1 else 'Input', markersize=2
              , markeredgewidth=1, alpha = 0.7
              , **cycle_opts)
        # Plot model outputs
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

  @property
  def has_train_ds(self):
    return hasattr(self,"train_df")

  @property
  def has_val_ds(self):
    return hasattr(self,"val_df")

  @property
  def has_test_ds(self):
    return hasattr(self,"test_df")

  def _cycle_slice(self, cycle_idx=0):
    end = -(self.n_cycles-(cycle_idx+1))*self._cycle_shift
    start = end - self._cycle_width
    return slice(start,end if end != 0 else None)

  def _cycle_indices(self, cycle_idx=0):
    return self.full_window_indices[self._cycle_slice(cycle_idx)]

  def _make_dataset( self, df, opts, cache_filepath, memory_cache = False):
    #start = datetime.datetime.now()
    #print("Building new dataset...")
    # This will not work if attempting to forecast the past:
    try:
      sklearn.utils.validation.check_is_fitted( self._pp )
    except sklearn.exceptions.NotFittedError:
      self._pp.fit(self.train_df)
    opts.set_unset_to_default( self, df )
    # NOTE This slice on features must be removed when adding support to labels
    data = self._pp.transform( df.loc[:,self._features].to_numpy(dtype=np.float32) )
    # Create time series windows
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=self._sequence_stride,
        shuffle=False,
        batch_size=tf.constant(1,dtype=tf.int64),)
    if cache_filepath: cache_filepath += '_win%d' % self.total_window_size
    if cache_filepath: cache_filepath += '_stride%d' % self._sequence_stride
    # XXX Hack to remove batching
    ds = ds._input_dataset
    # Cache just after the heavy windowing operation so that it is shared
    # across each subset, independently of the next configs
    if self._sample_from_marginals:
      if cache_filepath: cache_filepath += '_marginals'
      ds = self._pattern_sampler(ds)
    if bool(opts.take_n):
      if cache_filepath: cache_filepath += '_take%d' % opts.take_n
      ds = ds.take( opts.take_n )
    if cache_filepath:
      if cache_filepath not in self._cached_filepath_dict:
        mkdir_p(cache_filepath)
        ds = ds.cache( cache_filepath )
        self._cached_filepath_dict[cache_filepath] = ds
      else:
        ds = ds.cache()
        print("Warning: Caching on memory although specified to cache on disk.\nReason: Dataset at '%s' is already currently being cached." % cache_filepath )
    if bool(opts.shuffle):
      ds = ds.shuffle(**opts.shuffle_kwargs)
    # Split windows into cycles
    ds = self._nested_timeseries(
        start_index = 0,
        end_index = self.total_window_size,
        input_dataset = ds,
        sequence_length=self._past_window_size+self._cycle_shift,
        sequence_stride=self._cycle_shift)
    ds = ds.map(self._extract_data)
    if opts.batch_size is not None:
      ds = ds.batch(opts.batch_size, drop_remainder = opts.drop_remainder)
    if memory_cache:
      ds = ds.cache()
    #total_time = datetime.datetime.now() - start
    #print("Finished building dataset in %s." % total_time)
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
    if slices:
      return { "cycle" : cycle, "slices" : tuple(slices) }
    else:
      return cycle

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
    # TODO: Perharps it is possible to create the datasets and save them to
    # avoid computing the shuffling buffer each time
    n = len(df)
    full_train_frac = 1. - test_frac
    train_frac      = 1. - val_frac

    full_train_slice              = slice(0,int(n*full_train_frac))
    full_train_df                 = df[full_train_slice]
    n_tr                          = int(n*full_train_frac)
    train_slice                   = slice(0,int(n_tr*full_train_frac))
    self.train_df                 = full_train_df[train_slice]

    val_slice                     = slice(int(n_tr*train_frac),None)
    self.val_df                   = full_train_df[val_slice]
    if hasattr(self.val_df, "reset_index"):
      self.val_df.reset_index(drop = True,inplace=True)

    test_slice                    = slice(int(n*full_train_frac)+1,None)
    self.test_df                  = df[test_slice]
    if hasattr(self.test_df, "reset_index"):
      self.test_df.reset_index(drop = True,inplace=True)
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

