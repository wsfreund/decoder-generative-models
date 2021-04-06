from .meter_base import EffMeterBase
from .ais_base.ais import AIS
from ..misc import *

import tensorflow as tf
import numpy as np

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
    self.normalization_constant = tf.cast(tf.reduce_prod(self.wrapper.generator.output.shape[1:]), tf.float32 )

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
    self.prev_final_state_iter = None
    if self.ais_state_cachefile:
      final_state = []; lower_bound_logp = []
      if self.data_batch_counter == 0:
        self.writer = tf.io.TFRecordWriter(self.ais_state_cachefile + '_writing')
        if os.path.exists(self.ais_state_cachefile):
          prev_final_state_ds = self.final_state_dataset_reader(batch = self.subbatch_size)
          prev_final_state_ds = prev_final_state_ds.map(lambda x: x['final_state'])
          self.prev_final_state_iter = iter(prev_final_state_ds)
    subbatch_size = data.shape[0] if self.subbatch_size is None else self.subbatch_size
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.batch(subbatch_size)
    final_lld = tf.constant(0., dtype=tf.float32)
    for batch in ds:
      #prev_state = next(self.prev_final_state_iter) if self.prev_final_state_iter is not None else None
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
        # NOTE Run the gap only for the best reconstructed sample
        # FIXME
        #import pdb; pdb.set_trace()
        #best_final_state = l_final_state[:,np.argmax(l_avg_lower_bound_logp),...]
        best_final_state = np.take_along_axis(l_final_state.numpy(),np.expand_dims(np.argmax(l_lower_bound_logp,axis=0),axis=(0,2)),axis=0).squeeze()
        l_x = self.wrapper.generator( best_final_state ) # FIXME
        l_backward_lld, _, _, l_backward_avg_lower_bound_logp = self.backend.backward_ais(l_x, best_final_state )
        self.backward_counter += 1
        # Incremental average update
        self.backward_lld += ( l_backward_lld - self.backward_lld ) / self.backward_counter
        self.backward_avg_lower_bound_logp.append( l_backward_avg_lower_bound_logp )
    final_lld /= subbatch_size # average over subbatches
    self.final_lld += final_lld
    if self.ais_state_cachefile:
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

  def final_state_dataset_reader(self, batch = None, filepath = None):
    """
    If batch is provided, returns [chain,batch,latent],
    otherwise returns [1,chain,latent] for the final_state.
    """
    if filepath is None: filepath = self.ais_state_cachefile
    tfrecord_dataset = tf.data.TFRecordDataset([filepath])
    def read_tfrecord(serialized_example):
      feature_description = {
          'final_state': tf.io.FixedLenFeature((), tf.string),
          'lower_bound_logp': tf.io.FixedLenFeature((), tf.string),
      }
      example = tf.io.parse_single_example(serialized_example, feature_description)
      final_state = tf.io.parse_tensor(example['final_state'], out_type = tf.float32)
      final_state = tf.transpose(final_state, [1, 0, 2])
      lower_bound_logp = tf.io.parse_tensor(example['lower_bound_logp'], out_type = tf.float32)
      lower_bound_logp = tf.transpose(lower_bound_logp, [1, 0,])
      return { 'final_state' : final_state, 'lower_bound_logp' : lower_bound_logp }
    parsed_dataset = tfrecord_dataset.map(read_tfrecord)
    parsed_dataset = parsed_dataset.unbatch()
    if batch:
      def permute(d):
        return { k : tf.transpose(v,[1,0,2] if k == 'final_state' else [1,0]) for k, v in d.items() }
      parsed_dataset = parsed_dataset.batch(batch)
      parsed_dataset = parsed_dataset.map(permute)
    return parsed_dataset

  def serialize(self, final_state, lower_bound_logp):
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'final_state':      _bytes_feature(tf.io.serialize_tensor(final_state))
      , 'lower_bound_logp': _bytes_feature(tf.io.serialize_tensor(lower_bound_logp))
      }))
    self.writer.write(tf_example.SerializeToString())

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
