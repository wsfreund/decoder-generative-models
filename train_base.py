import os, sys
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tempfile
import itertools
import datetime
from tensorflow.keras import layers

try:
  from misc import *
  from mask_base import MaskModel
except ImportError:
  from .misc import *
  from .mask_base import MaskModel

class InterruptTraining( BaseException ):
  pass

class BreakDueToUpdates( InterruptTraining ):
  pass

class BreakDueToEpoches( InterruptTraining ):
  pass

class BreakDueToMaxFail( InterruptTraining ):
  pass

class BreakDueToWallTime( InterruptTraining ):
  pass

class TrainingCriticalAbort( BaseException ):
  pass

class BreakDueToNonFinite( TrainingCriticalAbort ):
  pass

class TrainBase(MaskModel):
  def __init__(self, **kw):
    ## Configuration
    # tf_call_kw: propagate "training = True" for dropout
    self._tf_call_kw               = retrieve_kw(kw, 'tf_call_kw',           {}                                                         )
    # Cycle for computing validation steps
    self._validation_steps         = retrieve_kw(kw, 'validation_steps',     25                                                         )
    # Maximum number of training epoches (cycles through training dataset)
    self._max_epoches              = retrieve_kw(kw, 'max_epoches',          None                                                       )
    # Maximum number of parameter updates
    self._max_steps                = retrieve_kw(kw, 'max_steps',            None                                                       )
    # Maximum wall time
    self._max_train_wall_time      = retrieve_kw(kw, 'max_wall_time',        None                                                       )
    # Maximum number of fails to improve the validation criterion
    self._max_fail                 = retrieve_kw(kw, 'max_fail',             10000                                                      )
    # Minimum progress on the validation criterion to consider a valid progress
    self._min_progress             = retrieve_kw(kw, 'min_progress',         1e-4                                                       )
    # Whether to log training progress
    self._verbose                  = retrieve_kw(kw, 'verbose',              False                                                      )
    # Interval for logging using updates
    self._print_interval_updates   = retrieve_kw(kw, 'print_interval_updates', 1000                                                     )
    # Interval for logging using wall time
    self._print_interval_wall_time = retrieve_kw(kw, 'print_interval_wall_time', datetime.timedelta( seconds = 15 )                     )
    # Interval for logging using epoches
    self._print_interval_epoches   = retrieve_kw(kw, 'print_interval_epoches', 5                                                        )
    # File path to be used when saving/loading
    self._result_file              = retrieve_kw(kw, 'result_file',          "model.weights"                                            )
    # Batch size used for computations during training
    self._batch_size               = retrieve_kw(kw, 'batch_size',           128                                                        )
    # Batch size used for model evaluation using full dataset. When None, use the training size.
    self._eval_batch_size          = retrieve_kw(kw, 'eval_batch_size',      None                                                       )
    # Element-wise gradient clipping value
    self._grad_clipping            = tf.constant( retrieve_kw(kw, 'grad_clipping',        False  ), dtype=tf.float32 )
    # Shuffle buffer for the training dataset. When False, disable shuffling. When None, use the training size.
    self._shuffle_buffer           = retrieve_kw(kw, 'shuffle_buffer',       None                                                       )
    # Reshuffle dataset at each iteration
    self._reshuffle_each_iteration = retrieve_kw(kw, 'reshuffle_each_iteration', False                                                  )
    ## Setup
    self._lkeys      = set()
    self._val_lkeys  = set()
    self._val_prefix = '' # TODO Make it become a set
    self._loss_fcn = None
    self._model_dict = {}
    ## Sanity checks
    if self._max_train_wall_time is not None:
      assert isinstance(self._max_train_wall_time, datetime.timedelta)
    if self._print_interval_wall_time is not None:
      assert isinstance(self._print_interval_wall_time, datetime.timedelta)

  def train(self, train_data, train_mask = None, val_data = None, val_mask = None):
    start_train_wall_time = datetime.datetime.now()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    n_gpus = len(gpus)
    if self._verbose: print('This machine has %i GPUs.' % n_gpus)

    # NOTE without drop_remainder, there is the need to specify the input_signature, i.e.:
    # @tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.float32),))
    train_dataset = tf.data.Dataset.from_tensor_slices( (train_data, train_mask), )
    if self._shuffle_buffer is not False:
      train_dataset = train_dataset.shuffle( train_data.shape[0] if self._shuffle_buffer is None else self._shuffle_buffer
                                           , reshuffle_each_iteration=self._reshuffle_each_iteration )
    train_dataset = train_dataset.batch( self._batch_size, drop_remainder = True )
        
    determinist_train_dataset = (tf.Variable(train_data, dtype = tf.float32), tf.Variable(train_mask, dtype = tf.float32) if train_mask else None )

    if val_data is not None:
      if "early_stopping_key" not in self.__dict__:
        raise ValueError("Specified val_data but no early_stopping key is specified.")
      else:
        val_dataset = (tf.Variable(val_data, dtype = tf.float32), tf.Variable(val_mask, dtype = tf.float32) )
        best_model_temp_path = os.path.join( tempfile.mkdtemp(), os.path.basename(self._result_file) )
      best_epoch = 0; best_step = 0; last_progress_step = 0; best_val_reco = np.finfo( dtype = np.float32 ).max
    else:
      val_dataset = None

    if val_data is None and self._max_epoches is None and self._max_steps is None:
      raise ValueError("Stopping criteria not specified. Either specify early stopping, max number of epoches or steps or any combination of these criteria.")

    # containers for losses
    loss_record = {k : [] for k in self._lkeys}
    val_loss_record = {k : [] for k in self._val_lkeys}

    # TODO reduce_lr = ReduceLROnPlateau(monitor='loss', patience=100, mode='auto',
    #                              factor=factor, cooldown=0, min_lr=1e-4, verbose=2)
    step = batches = last_cycle = 0
    skipFinalVal = is_new_cycle = False;
    exc_type = exc_val =  None
    try:
      for epoch, _ in enumerate(itertools.repeat(*[None]+([self._max_epoches] if self._max_epoches is not None else []))):
        alreadyPrintedEpoch = False
        for sample_batch, mask_batch in train_dataset:
          evaluatedVal = False
          loss_dict = self._train_base(epoch, step, sample_batch, mask_batch)
          loss_dict = self._parse_train_loss( loss_dict, self._val_prefix )
          # Keep track of training record:
          self._append_loss(loss_record, loss_dict)
          step += 1
          val_loss_dict = {}
          # Print logging information
          if val_dataset is not None and ( not(step % self._validation_steps) or step == 1 ) and self._loss_fcn:
            val_loss_dict = self._loss_fcn( val_dataset[0], val_dataset[1] )
            val_loss_dict['step'] = step
            evaluatedVal = True
            self._append_loss(val_loss_record, val_loss_dict, keys = self._val_lkeys)
            if val_loss_dict[self.early_stopping_key] < best_val_reco:
              if best_val_reco - val_loss_dict[self.early_stopping_key] > self._min_progress:
                last_progress_step = step
              best_val_reco = val_loss_dict[self.early_stopping_key]
              best_step = step; best_epoch = epoch 
              self.save( overwrite = True, output_file = best_model_temp_path )
            if ( step - last_progress_step ) >= self._max_fail:
              raise BreakDueToMaxFail()
          train_time = datetime.datetime.now() - start_train_wall_time
          cycle = int( train_time / self._print_interval_wall_time ) if self._print_interval_wall_time is not None else 0
          is_new_cycle = cycle > last_cycle
          if (self._verbose and 
                (
                  (not(step % self._print_interval_updates)) 
                  or (not(epoch % self._print_interval_epoches) and not(alreadyPrintedEpoch) )
                  or is_new_cycle
                )
              ):
            last_improvement = { 'best_val_reco' : best_val_reco
                , 'best_step' : best_step
                , 'last_progress_step' : last_progress_step } if val_loss_dict else {}
            self._print_progress(epoch, step, train_time, loss_dict, val_loss_dict, last_improvement )
            if not(epoch % self._print_interval_epoches):
              alreadyPrintedEpoch = True
            if is_new_cycle:
              last_cycle = cycle
              is_new_cycle = False
          if self._max_steps is not None and (step + 1 > self._max_steps):
            raise BreakDueToUpdates()
          if self._max_train_wall_time is not None and (train_time > self._max_train_wall_time):
            raise BreakDueToWallTime()
      raise BreakDueToEpoches
    except BaseException as e:
      exc_type, exc_val = sys.exc_info()[:2]
    finally:
      if isinstance( exc_val, InterruptTraining):
        print('Training finished!')
        interruptTraining = True
        if isinstance( exc_val, BreakDueToMaxFail ):
          # Recover best validation result
          print('Reason: early stopping.')
          print('Recovering Best Validation Performance @ (Epoch %i, Step %i).' % (best_epoch, best_step,))
          print('Reco_loss: %.3f.' % best_val_reco)
          self.load( best_model_temp_path )
          skipFinalVal = True
        elif isinstance( exc_val, BreakDueToUpdates ):
          print('Reason: max steps.')
        elif isinstance( exc_val, BreakDueToEpoches ):
          print('Reason: max epoches.')
        elif isinstance( exc_val, BreakDueToWallTime):
          print('Reason: reached wall time limit.')
      # Other non-critical interruptions
      elif isinstance( exc_val, KeyboardInterrupt):
        interruptTraining = True
        print('Training finished!')
        print('Reason: user interrupted training.')
      # Critical interruptions
      elif isinstance( exc_val, (TrainingCriticalAbort, BaseException) ):
        print('ERROR: Training aborted!')
        if isinstance( exc_val, BreakDueToNonFinite ):
          print('Reason: found non-finite value!!')
        raise exc_val
      if val_dataset is not None and self._loss_fcn:
        if not skipFinalVal:
          if not evaluatedVal:
            val_loss_dict = self._loss_fcn( val_dataset[0], val_dataset[1] )
            val_loss_dict['step'] = step
            self._append_loss(val_loss_record, val_loss_dict, keys = self._val_lkeys)
          if val_loss_dict[self.early_stopping_key] < best_val_reco:
            best_val_reco = val_loss_dict[self.early_stopping_key]
            best_step = step; best_epoch = epoch 
          else:
            print('Recovering Best Validation Performance @ (Epoch %i, Step %i).' % (best_epoch, best_step,))
            print('Reco_loss: %.3f.' % (best_val_reco))
            self.load( best_model_temp_path )
    self.save( overwrite = True )
    # Compute final performance:
    final_performance = {}
    if self._loss_fcn:
      final_performance['trn'] = self._loss_fcn( determinist_train_dataset[0] , determinist_train_dataset[1] )
      if val_dataset is not None:
        final_performance['val'] = self._loss_fcn( val_dataset[0], val_dataset[1] )
        final_performance['val']['best_step'] = best_step
        final_performance['val']['best_epoch'] = best_epoch
      else:
        final_performance['val'] = dict()
    return { 'train_record' : loss_record
           , 'val_record' : val_loss_record
           , 'final_performance' : final_performance }

  def loss_per_dataset(self, x, mask
      , x_val = None, mask_val = None 
      , x_tst = None, mask_tst = None
      , fcn = None):
    if fcn is None: fcn = self._loss_fcn
    return { 'trn' : fcn(x,mask)
           , 'val' : fcn(x_val,mask_val) if x_val is not None else {}
           , 'tst' : fcn(x_tst,mask_tst) if x_tst is not None else {} 
           }

  def save(self, overwrite = False, output_file = None ):
    if output_file is None:
      output_file = self._result_file
    for k, m in self._model_dict.items():
      m.save_weights( output_file + '_' + k, overwrite )

  def load(self, path):
    for k, m in self._model_dict.items():
      m.load_weights( path + '_' + k)

  def plot_model(self, model_name, *args, **kw):
    if model_name in self._model_dict:
      model = fix_model_layers( self._model_dict[model_name] )
      return tf.keras.utils.plot_model(model, *args, **kw)
    else:
      raise KeyError( "%s is not a valid model key. Available models are: %s" % (model_name, self._model_dict.keys()))

  def _append_loss(self, loss_record, loss_dict, keys = None):
    if keys is None:
      keys = self._lkeys
    for k in keys:
      if k in loss_dict:
        loss_record[k].append(loss_dict[k].numpy() if hasattr(loss_dict[k],"numpy") else loss_dict[k])
      else:
        loss_record[k].append(np.nan)

  def _accumulate_loss_dict( self, acc_dict, c_dict):
    for k in c_dict.keys():
      val = c_dict[k].numpy() if hasattr(c_dict[k],"numpy") else c_dict[k]
      if k in acc_dict:
        acc_dict[k] += val
      else:
        acc_dict[k] = val

  def _train_base(self, epoch, step, sample_batch, mask_batch):
    loss_dict = self._train_step(sample_batch, mask_batch)
    return loss_dict

  def _parse_train_loss(self, train_loss, prefix = None):
    if prefix and not(prefix.endswith('_')): prefix += '_'
    for k, v in train_loss.items():
      if prefix and not k.startswith( prefix ): continue
      if tf.math.logical_not(tf.math.is_finite(v)):
        raise BreakDueToNonFinite(k)
    return train_loss

  def _loss_fcn(self, x, mask):
    raise NotImplementedError("Overload loss function to whichever function computes the loss.")

  def _print_progress(self, epoch, step, train_time, loss_dict, val_loss_dict, last_improvement ):
    try:
      perc_epoches = np.around(100*epoch/self._max_epoches, decimals=1)
    except:
      perc_epoches = 0
    try:
      perc_steps = np.around(100*step/self._max_steps, decimals=1)
    except:
      perc_steps = 0
    try:
      perc_wall = np.around(100*train_time/self._max_train_wall_time, decimals=1)
    except:
      perc_wall = 0
    perc = max(perc_epoches, perc_steps, perc_wall)
    perc = min(perc,100.)
    perc_str = ("%2.1f%%" % perc) if perc >= 0. else '??.?%%'
    print(('Epoch: %i. Steps: %i. Time: %s. Training %s complete. ' % (epoch, step, train_time, perc_str)) + 
      '; '.join([("%s: %.3f" % (k, v)) for k, v in loss_dict.items()])
    )
    if val_loss_dict:
      print(('Validation (Step %i): ' % (val_loss_dict['step'])) +
        self.early_stopping_key + "_val" + (': %.3f; ' % (val_loss_dict[self.early_stopping_key])) +
        '; '.join([("%s: %.3f" % (k + "_val", v)) for k, v in val_loss_dict.items() if k not in ('step', self.early_stopping_key)])
      )
      delta = (step - last_improvement['last_progress_step']) if (step - last_improvement['last_progress_step']) > self._validation_steps else 0
      print( ( 'Best Val: %.3f (step=%d).'  % 
          ( last_improvement['best_val_reco']
          , last_improvement['best_step'] ) )
          + ( ( 
          ' Fails = [%d/%d]' % 
        ( delta
        , self._max_fail)
        ) if delta else "" )
      )

