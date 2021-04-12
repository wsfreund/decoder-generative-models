from .meter_base import EffMeterBase
from .ais_base.ais import AIS
from ..misc import *

import tensorflow as tf
import numpy as np
import itertools

class AISMeter(EffMeterBase):
  def __init__(self 
      , var
      , name = "ais"
      , nchains = 8
      , max_ais_batches = None
      , subbatch_size = None
      , axis = [-2, -1]
      , mcmc_args = {}
      , schedule_args = {'n_points' : 1001}
      , data_parser = lambda x: x
      , keep_avg_logp = False
      , run_backward_ais = False
      , ais_state_cachefile = '' ):
    super().__init__(name, data_parser)
    # configuration
    self.var = var
    self.nchains = nchains
    self.max_ais_batches = max_ais_batches
    self.subbatch_size = subbatch_size
    self.axis = axis
    self.mcmc_args = mcmc_args
    self.schedule_args = schedule_args
    self.keep_avg_logp = keep_avg_logp
    self.ais_state_cachefile = ais_state_cachefile
    self.run_backward_ais = run_backward_ais
    # runtime variables
    self.final_lld = 0.
    self.avg_lower_bound_logp = []
    # Used when running gap estimation
    self.backward_counter = 0
    self.backward_lld = 0.
    self.backward_avg_lower_bound_logp = []

  def initialize(self, wrapper):
    self.wrapper = wrapper
    self.normalization_constant = tf.cast(tf.reduce_prod(wrapper._transform_to_meter_input(self.wrapper.generator.output).shape[1:]), tf.float32 )

  @property
  def backend(self):
    # FIXME There is some issue when running multiple times the AIS graph. It
    # seems to be some incompatibility with keras
    # XXX Solution is to create an instance every time we run it
    ais = AIS( self.wrapper
             , schedule_args = self.schedule_args 
             , mcmc_args = self.mcmc_args
             , observation_model_args = dict( var = self.var, axis = self.axis )
             , nchains = self.nchains )
    return ais

  def update_on_parsed_data(self, data, mask = None):
    if mask is not None:
      raise NotImplementedError("%s is not currently implemented for masked data" % self.__class__.__name__)
    if self.max_ais_batches and self.data_batch_counter >= self.max_ais_batches:
      return
    subbatch_size = self.wrapper.get_batch_size_from_data(data) if self.subbatch_size is None else self.subbatch_size
    self.prev_final_state_iter = None
    if self.ais_state_cachefile:
      final_state = []; lower_bound_logp = []
      if self.data_batch_counter == 0:
        self.writer = tf.io.TFRecordWriter(self.ais_state_cachefile + '_writing')
        if os.path.exists(self.ais_state_cachefile):
          prev_final_state_ds = self.final_state_dataset_reader(batch = subbatch_size)
          # not to be used until the state_ds returns a list with the final_state
          # prev_final_state_ds = prev_final_state_ds.map(lambda x: x['final_state'])
          self.prev_final_state_iter = iter(prev_final_state_ds)
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.batch(subbatch_size)
    final_lld = tf.constant(0., dtype=tf.float32)
    for batch in ds:
      #prev_state = self._read_tf_record.to_list( next(self.prev_final_state_iter) )['final_state'] if self.prev_final_state_iter is not None else None
      prev_state = None
      if prev_state is not None: prev_state = tf.reshape(prev_state,[-1,self.wrapper.latent_dim()])
      l_final_lld, l_final_state, l_lower_bound_logp, l_avg_lower_bound_logp = self.backend.forward_ais(batch, prev_state)
      final_lld += l_final_lld 
      if self.ais_state_cachefile:
        final_state.append( l_final_state )
        lower_bound_logp.append( l_lower_bound_logp )
      if self.keep_avg_logp:
        self.avg_lower_bound_logp.append( l_avg_lower_bound_logp )
      if self.run_backward_ais:
        def take(data, idxs):
          idxs = np.expand_dims(idxs,axis=[0]+list(range(2,data.shape.rank)))
          return np.take_along_axis(data.numpy(),idxs,axis=0).squeeze()
        best_idxs = np.argmax(l_lower_bound_logp,axis=0)
        if isinstance(l_final_state, (tuple,list)):
          best_final_state = tuple(take(d, best_idxs) for d in l_final_state)
        else:
          best_final_state = take(l_final_state, best_idxs)
        l_x = self.wrapper.generator( best_final_state )
        l_backward_lld, _, _, l_backward_avg_lower_bound_logp = self.backend.backward_ais(l_x, best_final_state )
        self.backward_counter += 1
        # Incremental average update
        self.backward_lld += ( l_backward_lld - self.backward_lld ) / self.backward_counter
        self.backward_avg_lower_bound_logp.append( l_backward_avg_lower_bound_logp )
    final_lld /= subbatch_size # average over subbatches
    self.final_lld += final_lld
    if self.ais_state_cachefile:
      if isinstance(final_state[0], (tuple,list)):
        final_state = tuple(tf.concat(list(map(lambda c: c[i],final_state)), axis = 1) for i in range(len(final_state[0])))
      else:
        final_state = tf.concat( final_state, axis = 1)
      lower_bound_logp = tf.concat( lower_bound_logp, axis = 1)
      self.serialize(final_state, lower_bound_logp)

  def retrieve(self):
    self.start
    self.final_lld /= self.data_batch_counter
    if self.keep_avg_logp:
      self.avg_lower_bound_logp = tf.concat( self.avg_lower_bound_logp, axis = 1)
    avg_lower_bound_dict = {
        # self.name + "_std" : tf.math.reduce_std(self.avg_lower_bound_logp)
        self.name + "_std_norm" : tf.math.reduce_std(self.avg_lower_bound_logp) / self.normalization_constant
    } if self.keep_avg_logp else {}
    if self.ais_state_cachefile:
      self.writer.close()
      import shutil
      shutil.move(self.ais_state_cachefile + '_writing', self.ais_state_cachefile)
    backward_dict = {
        # self.name + "_gap" : self.backward_lld - self.final_lld
        self.name + "_gap_norm" : ( self.backward_lld - self.final_lld ) / self.normalization_constant
    } if self.run_backward_ais else {}
    if self.run_backward_ais:
      self.backward_avg_lower_bound_logp = tf.concat( self.backward_avg_lower_bound_logp, axis = 1)
    ret = { 'neg_' + self.name + "_norm" : - self.final_lld / self.normalization_constant }
    ret.update(avg_lower_bound_dict)
    ret.update(backward_dict)
    self.stop
    return ret

  def reset(self):
    self.final_lld = 0.
    self.avg_lower_bound_logp = []
    if self.run_backward_ais:
      self.backward_counter = 0
      self.backward_lld = 0.
      self.backward_avg_lower_bound_logp = []
    super().reset()

  def final_state_dataset_reader(self, batch = 1, filepath = None):
    """
    If batch is provided, returns [chain,batch,latent],
    otherwise returns [1,chain,latent] for the final_state.
    """
    # FIXME messy code
    if filepath is None: filepath = self.ais_state_cachefile
    tfrecord_dataset = tf.data.TFRecordDataset([filepath])
    if not hasattr(self,'_read_tf_record'):
      feature_description, perm_const = self.serialized_feature_description(tfrecord_dataset)
      self._read_tf_record = _Read_TFRecord( feature_description, perm_const ) 
    raw_example = next(iter(tfrecord_dataset))
    example = self._read_tf_record(raw_example)
    parsed_dataset = tfrecord_dataset.map(self._read_tf_record)
    parsed_dataset = parsed_dataset.unbatch()
    parsed_dataset = parsed_dataset.batch(batch)
    parsed_dataset = parsed_dataset.map(self._read_tf_record.permute)
    # TODO final_state_... /> final_state
    return parsed_dataset

  def serialized_feature_description(self, tfrecord_dataset):
    raw_example = next(iter(tfrecord_dataset))
    example = tf.train.Example()
    example.ParseFromString(raw_example.numpy())
    feature_description = {
      'lower_bound_logp': tf.io.FixedLenFeature((), tf.string),
    }
    feature_description.update({
      s : tf.io.FixedLenFeature((), tf.string) for s in example.features.feature.keys() if s.startswith('final_state')
    })
    example = tf.io.parse_single_example(raw_example, feature_description)
    perm_const = []
    for k in sorted(example.keys()):
      if k.startswith('final_state'):
        data = tf.io.parse_tensor(example[k], out_type=tf.float32)
        rank = data.shape.rank
        perm_const.append([1,0] + list(range(2,rank)))
    return feature_description, perm_const

  def serialize(self, final_state, lower_bound_logp):
    feature_dict = {
      'lower_bound_logp': _bytes_feature(lower_bound_logp)
    }
    if not isinstance(final_state,(tuple,list)):
      feature_dict.update({ 'final_state': _bytes_feature(final_state) })
    else:
      feature_dict.update({ ('final_state_%d' % i ): _bytes_feature(final_state[i]) for i in range(len(final_state))  })
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    self.writer.write(tf_example.SerializeToString())


def _int_feature(value):
  """Returns a int from a string / byte."""
  def parse(data):
    data = tf.io.serialize_tensor(data)
    if isinstance(data, type(tf.constant(0))):
      data = data.numpy() # BytesList won't unpack a string from an EagerTensor.
    return data
  #if isinstance(value,(tuple,list)):
  #  value = [parse(v) for v in value]
  #else:
  value = [parse(value)]
  return tf.train.Feature(bytes_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  def parse(data):
    data = tf.io.serialize_tensor(data)
    if isinstance(data, type(tf.constant(0))):
      data = data.numpy() # BytesList won't unpack a string from an EagerTensor.
    return data
  #if isinstance(value,(tuple,list)):
  #  value = [parse(v) for v in value]
  #else:
  value = [parse(value)]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

class _Read_TFRecord(object):
  # FIXME This implementation is really inefficient and difficult to read.

  def __init__(self, feature_description, perm_const):
    self.feature_description = feature_description
    self._list_to_single_object = 'final_state' in self.feature_description
    self.perm_const = perm_const
    self.axis_to_change = tf.constant([1,0], dtype=tf.int32)

  @tf.function
  def __call__(self,serialized_example):
    example = tf.io.parse_single_example(serialized_example, self.feature_description)
    example_dict = {}
    for k in sorted(example.keys()):
      if k.startswith('final_state'):
        example_dict[k] = tf.io.parse_tensor(example[k], out_type = tf.float32)
    example_dict['lower_bound_logp'] = tf.io.parse_tensor(example['lower_bound_logp'], out_type = tf.float32)
    return self.permute(example_dict)

  @tf.function
  def permute(self,example):
    example = example.copy()
    i = 0
    for k in example.keys():
      if k.startswith('final_state'):
        example[k] = tf.transpose(example[k], perm=self.perm_const[i])
        i += 1
      else:
        example[k] = tf.transpose(example[k], perm=self.axis_to_change)
    return example

  def to_list(self,example):
    final_state = []
    for k in sorted(example.keys()):
      if k.startswith('final_state'):
        final_state.append(example[k])
    if self._list_to_single_object:
      final_state = final_state[0]
    return { 'final_state' : final_state
           , 'lower_bound_logp' : example['lower_bound_logp'] }

    #if self._list_to_single_object:
    #  final_state = final_state[0]
