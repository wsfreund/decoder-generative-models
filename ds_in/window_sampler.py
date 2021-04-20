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
  Allows to specify different sampling options for each sampling type
  """

  def __init__( self, sequence_stride = NotSet, **kw ):
    self.sequence_stride = sequence_stride
    super().__init__(**kw)

  def set_unset_to_default(self, sampler, df):
    super().set_unset_to_default( sampler, df )
    if self.sequence_stride is NotSet:
      self.sequence_stride =  sampler._default_sequence_stride
    if not "buffer_size" in self.shuffle_kwargs:
      if sampler.shuffle_buffer_size_window is NotSet:
        shuffle_buffer_size_window = datetime.timedelta(days=TimeseriesMetadata.days_per_year)
      else:
        assert isinstance(sampler.shuffle_buffer_size_window, datetime.timedelta)
        shuffle_buffer_size_window = sampler.shuffle_buffer_size_window
      # Compute the buffer size
      val = int(np.round( shuffle_buffer_size_window
                        / (self.sequence_stride*sampler._time_step)
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
  """
  WindowSampler transforms dataframes with time-series data onto tensorflow datasets.
  Builds multiple datasets on-demand with different configurations specified by
  :class SpecificFlowWindowSamplingOpts:.

  WARNING: This code was not extensively debuged nor tested, so it must be used
  with caution.

  :param manager: A :class DataFrameManager: instance
  :param cycle_width: Number of time samples in the main window. The main
  window is on the 'cycle' keyword of the dataset returned dict.
  :param cycle_shift: Each window frame will be displaced by this param.
  :param n_cycles: The number of cycles to extract.
  :param specific_flow_sampling_opt_class:  A type inheriting from
  SpecificFlowWindowSamplingOpts.
  :param shuffle_buffer_size_window: The size of the shuffle buffer. If not
  set, will default to an year in samples.
  :param features: Specified which features to sample from.
  :param sample_from_marginals: Sample from the marginals at random.
  :param keep_marginal_axis: Whether to keep the pattern dimension when
  sampling from marginals.  :param default_sequence_stride: Stride value to be
  used when not specified in surrogate/perf dicts. The stride specifies the
  displacement between each sampled window in the original dataframe. I.e. if
  stride is the same as the window size, then there is no overlap between every
  sample in the dataset.
  :param keep_marginal_label: If set to True, adds a keyword on the dict with
  the corresponding marginal from which the data was sampled from.
  :param past_widths: A list with the number of time samples to extract from
  the past with respect to the current data cycle. 
  The list must be specified in counterclockwise order.
  These are returned on the datasets on the 'shift' keyword of the dict.
  :param past_widths_wrt_present: If set to True, the past_widths samples will
  mark the positions on the past with respect to the initial cycle time stamp.
  Otherwise, the first sample start specifying with respect to the initial
  cycle time stamp and then the next are with respect to the end of this
  window.
  For instance, suppose we want to extract two windows of a day of the past.
  We can achieve this by setting past_widths = [2 days in samples, 1 day in
  samples] when past_widths_wrt_present is set to true; or 
  past_widths = [1 day in samples, 1 day in samples] when
  past_widths_wrt_present is set to false.
  :param overlaps: The overlaps are a list of size equal to past_widths, each
  overlap brings the past window end forward for the specified number of time
  samples, causing it to overlap with the next window.
  Note that the overlap on the main cycles are specified through the
  cycle_shift. The overlap mode define the behavior w.r.t. the total past window
  sizes.
  The list must be specified in counterclockwise order.
  :param overlap_mode: A string defining how the overlaps and past_widths
  should be composed together to define the past windows.
    * 'conservative' (default): Keep the original past_width window size.
    * 'additive': The overlaps are added to the past_widths, which may result
    in larger windows than those defined in past_widths. When using this approach,
    the past_widths specify the number of fresh samples in the past windows, and
    the window size is the value specified on the past windows added to the
    overlaps.
  :param keep_timestamps: Returns the timestamps of the samples in the dict.
  :param add_periodic_info: Adds a keyword on the sampled dict with sin/cos
  samples with the specified periods [secs].
  """

  def __init__(self, manager, cycle_width, cycle_shift, **kw ):
    if "specific_flow_sampling_opt_class" not in kw:
      kw["specific_flow_sampling_opt_class"] = SpecificFlowWindowSamplingOpts
    # Strip information from dataset
    self._date_time = manager.date_time
    # NOTE _timestamps units are in seconds
    self._timestamps = tf.constant( self._date_time.map(datetime.datetime.timestamp).to_numpy(), dtype=tf.float32 )
    self._time_step = manager.time_step
    super().__init__( raw_data = manager.df, **kw)
    self._pp                        = manager.pre_proc
    self.shuffle_buffer_size_window = retrieve_kw(kw, "shuffle_buffer_size_window" )
    self._sample_from_marginals     = retrieve_kw(kw, "sample_from_marginals",      False                                )
    self._keep_marginal_axis        = retrieve_kw(kw, "keep_marginal_axis",         True                                 )
    self._default_sequence_stride   = retrieve_kw(kw, "default_sequence_stride",    1                                    )
    past_widths                     = retrieve_kw(kw, "past_widths",                None                                 )
    self._past_widths_wrt_present   = retrieve_kw(kw, "past_widths_wrt_present",    True                                 )
    self._overlaps                  = retrieve_kw(kw, "overlaps",                   None                                 )
    self._overlap_mode              = retrieve_kw(kw, "overlap_mode",               "conservative"                       )
    self._keep_timestamps           = retrieve_kw(kw, "keep_timestamps",            True                                 )
    # Add periodic information using periods specified in seconds
    self._add_periodic_info         = retrieve_kw(kw, "add_periodic_info",          None                                 )
    self._keep_marginal_labels      = retrieve_kw(kw, "keep_marginal_label",        False                                )
    features                        = retrieve_kw(kw, "features",                   []                                   )
    n_cycles                        = retrieve_kw(kw, "n_cycles",                   1                                    )
    # backward compatibility
    if "sequence_stride" in kw:
      self._default_sequence_stride   = kw["sequence_stride"]
    del kw

    if self._add_periodic_info and not isinstance(self._add_periodic_info,(tuple,list)):
      self._add_periodic_info = [self._add_periodic_info]

    # Work out the features
    self._features = features if features else list(filter(lambda s: s != 'timestamps', manager.df.columns.to_numpy().tolist()))
    #self.labels = labels if labels != slice(None) else manager.df.columns.to_numpy()
    self._feature_column_map = {l: i for i, l in enumerate(self._features)}
    #self.label_column_map = {l: i for i, l in enumerate(self.labels)}
    self._column_map = {name: i for i, name in enumerate(manager.df.columns)}

    # Work out the window parameters
    if past_widths is None:
      self._past_widths = past_widths
      self._past_window_size = 0
      self._past_widths_up_to_mark_zero = []
    else: # past_widths is not None
      if self._overlaps is None:
        self._overlaps = [0]*len(past_widths)
      if not isinstance(self._overlaps, np.ndarray):
        self._overlaps = np.array(self._overlaps, dtype=np.int_)
      if self._past_widths_wrt_present:
        self._past_window_size = max(past_widths)
        self._past_widths = np.array(past_widths) - np.array([0] + past_widths[:-1])
        self._past_widths_up_to_mark_zero = np.array(past_widths)
      else:
        self._past_window_size = sum(past_widths)
        self._past_widths = np.array(past_widths)
        self._past_widths_up_to_mark_zero = list(np.cumsum(past_widths, dtype=np.int64))
      if self._overlap_mode == "conservative":
        self._past_widths -= self._overlaps
        self._past_widths_up_to_mark_zero -= np.cumsum(self._overlaps) if self._past_widths_wrt_present else self._overlaps
        self._past_window_size -= sum(self._overlaps) 
      elif self._overlap_mode == "additive":
        pass
      else:
        raise RuntimeError("Unknown overlap_mode '%s'" % self._overlap_mode)
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
    # FIXME
    _CacheStorage.clear_cached_functions()
    SamplerBase.clear_cache(self)

  def update_n_cycles(self, n_cycles):
    # Remove all cached information in order to recompute datasets
    _CacheStorage.clear_cached_functions()
    # Compute windows
    self.n_cycles = n_cycles
    self.present_mark_step = self._past_window_size if self._past_widths is not None else 0
    self.future_window_size = (n_cycles-1)*self._cycle_shift + self._cycle_width
    self.total_window_size  = self._past_window_size + self.future_window_size

    # Compute slices
    self.full_past_slice = slice(0, self.total_window_size)
    self.full_window_indices = np.arange(self.total_window_size)
    self.sliding_window_cycle_slice = slice(-self._cycle_width,None)
    if self._past_widths is not None:
      if self._past_widths_wrt_present:
        edges = np.concatenate(
          [[0]
          , np.flip(np.abs(np.array([0] + self._past_widths_up_to_mark_zero[:-1].tolist()) - np.ones_like(self._past_widths_up_to_mark_zero)*self._past_window_size))
          ],axis=0
        )
      else:
        edges = np.cumsum([0] + list(reversed(self._past_widths)))
      self.all_past_slices = [slice(int(wstart), int(wend+g)) for wstart, wend, g in 
          # NOTE We reverse the edges twice because it is easier to implement the algorithm
          # in reverse order, but it is better to keep the input slices on the
          # original order
          zip( reversed(edges[:-1])
             , reversed(edges[1:])
             , self._overlaps)] # NOTE The overlaps are not reversed, as opposed to the edges
      self.all_past_indices = [self.full_window_indices[s] for s in self.all_past_slices]
    else:
      self.all_past_slices = []
      self.all_past_indices = []


  def plot(self, data_samples = None, plot_col = None, model = None,  ds = "val", n_examples = 3, do_display = False):
    from cycler import cycler
    import matplotlib.pyplot as plt
    import seaborn as sns
    from IPython.display import display
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
    fig, axarr = plt.subplots(nrows=n_examples, ncols=1, sharex=True, )
    if data_samples is not None:
      if 'data' in data_samples:
        data_samples = data_samples['data']
    for i in range(n_examples):
      ax = axarr[i]
      if data_samples is None:
        inputs = self.sample( ds = ds )
        if 'data' in inputs:
          inputs = inputs['data']
      else:
        inputs = { k : (v[i,...] if not isinstance(v,(tuple,list)) else tuple(vi[i,...] for vi in v)) for k, v in data_samples.items() }
      # Plot slices
      if feature_plot_col_index is not None or self._sample_from_marginals:
        slices = [d[:,:,feature_plot_col_index] for d in inputs["slices"]] if isinstance(inputs, dict) else []
        for j, (idxs, s, input_opts) in enumerate(zip(self.all_past_indices, slices, input_cycler)):
          ax.plot(delta[idxs], np.squeeze(s[0,:,:])#self._pp.inverse_transform(s)[0,:,:])
              , label='Past period [%d]' % j, markersize=2
              , markeredgewidth=1, **input_opts, alpha = 0.7)
      # Plot cycles
      for c, cycle_opts in zip(range(self.n_cycles), cycle_cycler):
        cycle = inputs["cycle"][c,:,feature_plot_col_index] if isinstance(inputs, dict) else inputs
        if plot_col: ax.set_ylabel(f'{plot_col}')
        if feature_plot_col_index is not None or self._sample_from_marginals:
          ax.plot(delta[self._cycle_indices(c)], np.squeeze(cycle)#self._pp.inverse_transform(cycle))
              , label=(('Cycle [%d]' % c) if c in (0,self.n_cycles-1) else '') if self.n_cycles>1 else 'Input', markersize=2
              , markeredgewidth=1, alpha = 0.7
              , **cycle_opts)
        # Plot model outputs
        if model is not None:
          # TODO
          outputs = model(inputs)
          ax.scatter(self.input_indices, np.squeeze(outputs[:,:,feature_plot_col_index]) # label_indices, feature_plot_col_index
            , label='Output' , marker = 'x' )
                        #marker='X', edgecolors='k', label='Predictions',
                        #c='#ff7f0e', s=64)
        if i == 0:
          ax.legend(loc=2)
    plt.xlabel('Time [%s]' % self._date_time.freq)
    if do_display: display(fig)
    return

  def _cycle_slice(self, cycle_idx=0):
    end = -(self.n_cycles-(cycle_idx+1))*self._cycle_shift
    start = end - self._cycle_width
    return slice(start,end if end != 0 else None)

  def _cycle_indices(self, cycle_idx=0):
    return self.full_window_indices[self._cycle_slice(cycle_idx)]

  def _make_dataset( self, df, opts, cache_filepath):
    #start = datetime.datetime.now()
    #print("Building new dataset...")
    # This will not work if attempting to forecast the past:
    try:
      sklearn.utils.validation.check_is_fitted( self._pp )
    except sklearn.exceptions.NotFittedError:
      self._pp.fit(self.raw_train_data)
    opts.set_unset_to_default( self, df )
    # NOTE This slice on features must be removed when adding support to labels
    features = self._features
    if self._keep_timestamps or self._add_periodic_info:
      # NOTE We will keep timestamp information as an additional feature until
      # we parse information with self._extract_data
      if not 'timestamps' in features:
        features.append('timestamps')
    data = self._pp.transform( df.loc[:,features].to_numpy(dtype=np.float32) )
    # Create time series windows
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=opts.sequence_stride,
        shuffle=False,
        batch_size=tf.constant(1,dtype=tf.int64),)
    if cache_filepath: cache_filepath += '_win%d' % self.total_window_size
    if cache_filepath: cache_filepath += '_stride%d' % opts.sequence_stride
    # XXX Hack to remove batching
    ds = ds._input_dataset
    # Cache just after the heavy windowing operation so that it is shared
    # across each subset, independently of the next configs
    if self._sample_from_marginals:
      if cache_filepath: cache_filepath += '_marginals'
      if cache_filepath and self._keep_marginal_labels: cache_filepath += '_withlabels'
      ds = self._pattern_sampler(ds)
    else:
      ds = ds.map(lambda x: {'data' : x})
    if cache_filepath and (self._keep_timestamps or self._add_periodic_info): cache_filepath += '_withtimestamps'
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
        sequence_length = self._past_window_size+self._cycle_width,
        sequence_stride = self._cycle_shift)
    ds = ds.map(self._extract_data)
    if opts.batch_size is not None:
      ds = ds.batch(opts.batch_size, drop_remainder = opts.drop_remainder)
    if opts.memory_cache:
      ds = ds.cache()
    #total_time = datetime.datetime.now() - start
    #print("Finished building dataset in %s." % total_time)
    return ds

  def _extract_data(self, data_dict):
    # Slice
    slices = []
    slices_timestamps = []
    all_data_window = data_dict['data']
    # Past slices 
    for s in self.all_past_slices:
      data = all_data_window[:,s,...]
      if self._keep_timestamps or self._add_periodic_info:
        timestamp = data[...,-1]
        data = data[...,:-1]
      if self._sample_from_marginals and not self._keep_marginal_axis:
        data = data[...,0]
      slices.append( data ) 
      if self._keep_timestamps or self._add_periodic_info: slices_timestamps.append( timestamp )
    cycle = all_data_window[:,self.sliding_window_cycle_slice,...]
    if self._keep_timestamps or self._add_periodic_info:
      cycle_timestamp = cycle[...,-1]
      cycle = cycle[...,:-1]
    cycle_info = {'cycle' : cycle} # Main cycle (present) information
    if slices: # Add past information, in sync with main cycle
      cycle_info.update({"slices" : tuple(slices)})
    if self._sample_from_marginals and self._keep_marginal_labels:
      ret_dict = { 'data' : cycle_info
                 , 'pattern_idx' : tf.gather(data_dict['pattern_idx'], indices = 0)}
    elif self._keep_timestamps or self._add_periodic_info:
      ret_dict = {'data' : cycle_info }
    else:
      ret_dict = cycle_info
    if self._keep_timestamps:
      timestamp_dict = {'cycle': cycle_timestamp}
      if slices_timestamps:
        timestamp_dict['slices'] = tuple(slices_timestamps)
      ret_dict['timestamps'] = timestamp_dict
    if self._add_periodic_info:
      cycle_periodic_info = []
      for period in self._add_periodic_info:
        cycle_periodic_info.append( tf.math.sin((2 * np.pi / period) * cycle_timestamp ) )
        cycle_periodic_info.append( tf.math.cos((2 * np.pi / period) * cycle_timestamp ) )
      cycle_periodic_info = tf.stack( cycle_periodic_info, axis = -1 )
      if slices:
        slice_periodic_info = []
        for t in slices_timestamps:
          lperiodic = []
          for period in self._add_periodic_info:
            lperiodic.append( tf.math.sin((2 * np.pi / period) * cycle_timestamp ) )
            lperiodic.append( tf.math.cos((2 * np.pi / period) * cycle_timestamp ) )
          lperiodic = tf.stack( lperiodic, axis = -1 )
          slice_periodic_info.append(lperiodic)
      periodic_info_dict = {'cycle' : cycle_periodic_info}
      if slices:
        periodic_info_dict['slices'] = slice_periodic_info
      ret_dict['periodic'] = periodic_info_dict
    #if self._features is not None: # Apply slicing on window
    #  inputs = tf.stack(
    #      [inputs[:, self._column_map[name]] for name in self._features],
    #      axis=-1)
    ##inputs = tf.expand_dims( inputs, axis = 1 )
    #inputs.set_shape([self.n_cycles, self._past_widths, None])
    return ret_dict

  def _pattern_sampler(self, input_dataset):
    from tensorflow.python.data.ops import dataset_ops
    from tensorflow.python.ops import array_ops
    from tensorflow.python.ops import math_ops
    # Determine the lowest dtype to store start positions (to lower memory usage).
    num_seqs = len(self._features)
    if self._keep_timestamps or self._add_periodic_info:
      # If running timestamps, we have an additional 'feature' to
      #keep this information
      num_seqs -= 1 
    index_dtype = 'int32' if num_seqs < 2147483647 else 'int64'

    positions = np.arange(0, num_seqs, dtype=index_dtype)
    feature_selection_ds = dataset_ops.Dataset.from_tensor_slices(positions)
    def cycle_select(array):
      def select(steps, pat): 
        # TODO return the pattern
        if self._keep_timestamps or self._add_periodic_info:
          data = tf.gather(steps, [pat,num_seqs], axis = -1)
        else:
          data = steps[:,pat:pat+1] if self._keep_marginal_axis else steps[:,pat]
        return { 'data' :  data
               , 'pattern_idx' : pat } 
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
        (dataset_ops.Dataset.range(len(start_positions)), positions_ds)
    ).map( indices_fcn)
    # TODO Add a dataset with the timestamps
    def cycle_data(array_dict):
      def retrieve_data(array_dict):
        ret_dict = { 'data' : 
            (
              array_dict['data'][start_index:end_index,:] 
                if not self._sample_from_marginals else 
              array_dict['data'][start_index:end_index]
            ) }
        if self._sample_from_marginals:
          ret_dict.update({'pattern_idx' : array_dict['pattern_idx']}) 
        return ret_dict
      def gather(steps, inds): 
        ret = { 'data' : array_ops.gather(steps['data'], inds, axis = 0) }
        # FIXME Using interleave will increase memory usage by copying pattern_idx per cycle.
        if self._sample_from_marginals:
          ret.update({'pattern_idx' : steps['pattern_idx']})
        return ret
      return dataset_ops.Dataset.zip(
           ( dataset_ops.Dataset.from_tensors( retrieve_data(array_dict) ).repeat(), indices_ds )
          ).map(gather).batch(self.n_cycles, drop_remainder = True)
    # Note that we are using a nested dataset structure
    dataset = input_dataset.interleave(
        cycle_data
      , cycle_length = 1
      , block_length = 1
    )
    return dataset

  def _split_data(self, df, val_frac, test_frac, **split_kw ):
    # TODO Add compatibility with time-series cross-validation
    # Note that we split without shuffling.
    # The goal is to have realistic generalization conditions, i.e. generalize
    # for future data
    # NOTE: If we are going to generalize for future and past data, this needs
    # to be taken into account.
    # TODO: Perharps it is possible to create the datasets and save them to
    # avoid computing the shuffling buffer each time
    df = df.copy()
    df['timestamps'] = self._timestamps
    n = len(df)
    full_train_frac = 1. - test_frac
    train_frac      = 1. - val_frac

    full_train_slice              = slice(0,int(n*full_train_frac))
    full_train_df                 = df[full_train_slice]
    n_tr                          = int(n*full_train_frac)
    train_slice                   = slice(0,int(n_tr*full_train_frac))
    self.raw_train_data           = full_train_df[train_slice]

    val_slice                     = slice(int(n_tr*train_frac),None)
    self.raw_val_data             = full_train_df[val_slice]
    if hasattr(self.raw_val_data, "reset_index"):
      self.raw_val_data.reset_index(drop = True,inplace=True)

    test_slice                    = slice(int(n*full_train_frac)+1,None)
    self.raw_test_data            = df[test_slice]
    if hasattr(self.raw_test_data, "reset_index"):
      self.raw_test_data.reset_index(drop = True,inplace=True)
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

