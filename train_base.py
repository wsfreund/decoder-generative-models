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

class Container(object):
  pass

class TrainBase(MaskModel):
  def __init__(self, data_sampler, **kw):
    super().__init__()
    self.data_sampler = data_sampler
    ## Configuration
    # Function to extract information from samples
    self.sample_parser_fcn         = retrieve_kw(kw,  'sample_parser_fcn',    None                                                      )
    # training_kw: propagate "training = True" for dropout
    self._training_kw              = retrieve_kw(kw, 'training_kw',          {'training' : True}                                        )
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
    # Whether to show online train plot
    self._load_model_at_path       = retrieve_kw(kw, 'load_model_at_path',   None                                                       )
    # Whether to show online train plot
    self._online_train_plot        = retrieve_kw(kw, 'online_train_plot',    False                                                      )
    # Interval for logging using updates
    self._print_interval_steps     = retrieve_kw(kw, 'print_interval_steps', 1000                                                       )
    # Interval for logging using wall time
    self._print_interval_wall_time = retrieve_kw(kw, 'print_interval_wall_time', datetime.timedelta( seconds = 15 )                     )
    # Interval for logging using epoches
    self._print_interval_epoches   = retrieve_kw(kw, 'print_interval_epoches', 5                                                        )
    # Interval for logging using updates
    self._save_interval_steps     = retrieve_kw(kw, 'save_interval_steps',   None                                                       )
    # Interval for logging using wall time
    self._save_interval_wall_time = retrieve_kw(kw, 'save_interval_wall_time', datetime.timedelta( minutes = 5 )                        )
    # Interval for logging using epoches
    self._save_interval_epoches   = retrieve_kw(kw, 'save_interval_epoches',  None                                                      )
    # Use log-sampling periods of loss functions
    self._use_log_history          = retrieve_kw(kw, 'use_log_history',      True                                                       )
    # Number of history samples when using log-sampled history
    self._max_n_history_samples    = retrieve_kw(kw, 'log_n_linear_history_samples',    50                                              )
    # Log-sampling period. To sample more, use lower values, i.e. 0.005. To sample less, higher values, i.e. 0.05
    self._log_sampling_period      = retrieve_kw(kw, 'log_sampling_period',  0.01                                                       )
    # File path to be used when saving/loading
    self._save_model_at_path       = retrieve_kw(kw, 'save_model_at_path',   "trained_model"                                            )
    # Batch size used for model evaluation using full dataset. When None, use the full dataset size.
    self._eval_batch_size          = retrieve_kw(kw, 'eval_batch_size',      None                                                       )
    # Whether to apply gradient clipping
    self._use_grad_clipping        = tf.constant( retrieve_kw(kw, 'use_grad_clipping', False  ) )
    # Gradient clipping function
    self._grad_clipping_fcn        = retrieve_kw(kw, 'grad_clipping_fcn', lambda x: tf.clip_by_norm( x, 2.0 )  )
    ## Setup
    self._lkeys      = {"step",}
    self._val_lkeys  = {"step",}
    self._val_prefix = '' # TODO Make it become a set
    self._loss_fcn = None
    self._model_dict = {}
    self._optimizer_dict = {}
    ## Sanity checks
    if self._max_train_wall_time is not None:
      assert isinstance(self._max_train_wall_time, datetime.timedelta)
    if self._print_interval_wall_time is not None:
      assert isinstance(self._print_interval_wall_time, datetime.timedelta)
    #if self._load_model_at_path is not None:

  def train(self):
    start_train_wall_time = datetime.datetime.now()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    n_gpus = len(gpus)
    if self._verbose: print('This machine has %i GPUs.' % n_gpus)

    # containers for losses
    lc = Container()
    if self._load_model_at_path is not None:
      lc.__dict__.update( self.load( self._load_model_at_path, return_loss = True) )
      # Load local_keys
      lc.__dict__.update( self.load( self._load_model_at_path, return_locals = True) )
    else:
      lc.loss_record = {k : [] for k in self._lkeys}
      lc.val_loss_record = {k : [] for k in self._val_lkeys}
      lc.best_epoch = lc.best_step = lc.last_progress_step = 0; lc.best_val_reco = np.finfo( dtype = np.float32 ).max
      lc.p_sample_period = -np.inf; lc.n_history_samples = 0;
      lc.step = lc.epoch = 0
    last_print_cycle = last_save_cycle = 0
    skipFinalVal = is_new_print_cycle = False;
    exc_type = exc_val =  None

    # TODO reduce_lr = ReduceLROnPlateau(monitor='loss', patience=100, mode='auto',
    #                              factor=factor, cooldown=0, min_lr=1e-4, verbose=2)
    try:
      while (lc.epoch < self._max_epoches if self._max_epoches else True):
        alreadyPrintedEpoch = alreadySavedEpoch = False
        for sample_batch in self.data_sampler.sampler_from_train_ds:
          if self.sample_parser_fcn is not None:
            sample_batch = self.sample_parser_fcn(sample_batch)
          evaluatedVal = False
          loss_dict = self._train_base(lc.epoch, lc.step, sample_batch)
          loss_dict = self._parse_train_loss( loss_dict, self._val_prefix )
          loss_dict['step'] = lc.step
          # Keep track of training record:
          c_sample_period = np.log10( lc.step + 1 ) // self._log_sampling_period
          if c_sample_period  > lc.p_sample_period:
            lc.p_sample_period = c_sample_period
            lc.n_history_samples = 0
          if lc.n_history_samples < self._max_n_history_samples:
            self._append_loss(lc.loss_record, loss_dict)
          val_loss_dict = {}
          # Print logging information
          if self.data_sampler.sampler_from_val_ds is not None and ( not(lc.step % self._validation_steps) or lc.step == 0 ) and self._loss_fcn:
            val_loss_dict = self._loss_fcn( self.data_sampler.sampler_from_val_ds )
            val_loss_dict['step'] = lc.step
            evaluatedVal = True
            if lc.n_history_samples < self._max_n_history_samples:
              self._append_loss(lc.val_loss_record, val_loss_dict, keys = self._val_lkeys)
            if val_loss_dict[self.early_stopping_key] < lc.best_val_reco:
              if lc.best_val_reco - val_loss_dict[self.early_stopping_key] > self._min_progress:
                lc.last_progress_step = lc.step
              lc.best_val_reco = val_loss_dict[self.early_stopping_key]
              lc.best_step = lc.step; lc.best_epoch = lc.epoch 
              self.save( overwrite = True, val = True )
            if ( lc.step - lc.last_progress_step ) >= self._max_fail:
              raise BreakDueToMaxFail()
          train_time = datetime.datetime.now() - start_train_wall_time
          print_cycle = int( train_time / self._print_interval_wall_time ) if self._print_interval_wall_time is not None else 0
          is_new_print_cycle = print_cycle > last_print_cycle
          # Print loss
          if ((self._verbose or self._online_train_plot) and 
                (
                  (not(lc.step % self._print_interval_steps) if self._print_interval_steps is not None else False) 
                  or ((not(lc.epoch % self._print_interval_epoches)  if self._print_interval_epoches is not None else False) and not(alreadyPrintedEpoch) )
                  or is_new_print_cycle
                )
              ):
            last_improvement = { 'best_val_reco' : lc.best_val_reco
                               , 'best_step' : lc.best_step
                               , 'last_progress_step' : lc.last_progress_step } if val_loss_dict else {}
            if self._online_train_plot:
              self._plot_train( lc.loss_record )
            if self._verbose: 
              self._replace_nans_with_last_report( loss_dict, lc.loss_record )
              self._print_progress(lc.epoch, lc.step, train_time, loss_dict, val_loss_dict, last_improvement )
            if not(lc.epoch % self._print_interval_epoches) if self._print_interval_epoches is not None else False:
              alreadyPrintedEpoch = True
            if is_new_print_cycle:
              last_print_cycle = print_cycle
              is_new_print_cycle = False
          if self._max_steps is not None and (lc.step + 1 > self._max_steps):
            raise BreakDueToUpdates()
          if self._max_train_wall_time is not None and (train_time > self._max_train_wall_time):
            raise BreakDueToWallTime()
          lc.step += 1
          if lc.n_history_samples < self._max_n_history_samples:
            lc.n_history_samples += 1
          save_cycle = int( train_time / self._save_interval_wall_time ) if self._save_interval_wall_time is not None else 0
          is_new_save_cycle = save_cycle > last_save_cycle
          # Save current model
          if (
              (not(lc.step % self._save_interval_steps) if self._save_interval_steps is not None else False) 
              or ((not(lc.epoch % self._save_interval_epoches)  if self._save_interval_epoches is not None else False) and not(alreadySavedEpoch) )
              or is_new_save_cycle
             ):
            loss_data = { 'train_record' : lc.loss_record
                        , 'val_record' : lc.val_loss_record }
            self.save( overwrite = True
                , loss_data = loss_data
                , locals_data = lc )
            if not(lc.epoch % self._save_interval_epoches) if self._save_interval_epoches is not None else False:
              alreadySavedEpoch = True
            if is_new_save_cycle:
              last_save_cycle = save_cycle
              is_new_save_cycle = False
        lc.epoch += 1
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
          print('Recovering Best Validation Performance @ (Epoch %i, Step %i).' % (lc.best_epoch, best_step,))
          print('Reco_loss: %.3f.' % lc.best_val_reco)
          self.load( self._save_model_at_path, val = True )
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
      if self.data_sampler.sampler_from_val_ds is not None and self._loss_fcn:
        if not skipFinalVal:
          if not evaluatedVal:
            val_loss_dict = self._loss_fcn( self.data_sampler.val )
            val_loss_dict['step'] = lc.step
            self._append_loss(lc.val_loss_record, val_loss_dict, keys = self._val_lkeys)
          if val_loss_dict[self.early_stopping_key] < lc.best_val_reco:
            lc.best_val_reco = val_loss_dict[self.early_stopping_key]
            best_step = lc.step; lc.best_epoch = lc.epoch 
          else:
            print('Recovering Best Validation Performance @ (Epoch %i, Step %i).' % (lc.best_epoch, best_step,))
            print('Reco_loss: %.3f.' % (lc.best_val_reco))
            self.load(  self._save_model_at_path, val = True )
    self.save( overwrite = True
             , locals_data = lc )
    # Compute final performance:
    final_performance = {}
    if self._loss_fcn:
      # FIXME determinist
      final_performance['trn'] = self._loss_fcn( determinist_train_dataset[0] , determinist_train_dataset[1] )
      if self.data_sampler.val is not None:
        final_performance['val'] = self._loss_fcn( self.data_sampler.sampler_from_val_ds )
        final_performance['val']['best_step'] = best_step
        final_performance['val']['best_epoch'] = lc.best_epoch
      else:
        final_performance['val'] = dict()
    loss_data = { 'train_record' : lc.loss_record
                , 'val_record' : lc.val_loss_record
                , 'final_performance' : final_performance }
    self.save( save_models_and_optimizers = False
             , loss_data = loss_data)
    return loss_data

  def loss_per_dataset(self, x, mask
      , x_val = None, mask_val = None 
      , x_tst = None, mask_tst = None
      , fcn = None):
    if fcn is None: fcn = self._loss_fcn
    return { 'trn' : fcn(x,mask)
           , 'val' : fcn(x_val,mask_val) if x_val is not None else {}
           , 'tst' : fcn(x_tst,mask_tst) if x_tst is not None else {} 
           }

  def save(self, overwrite = False, save_models_and_optimizers = True, val = False, loss_data = None, locals_data = None ):
    # Create folder if it does not exist
    if not os.path.exists(self._save_model_at_path):
      os.makedirs(self._save_model_at_path)
    if save_models_and_optimizers:
      for k, m in self._model_dict.items():
        if val: k += '_bestval'
        k +=  '.npz'
        self._save_model( os.path.join(self._save_model_at_path, k), m )
      for k, m in self._optimizer_dict.items():
        k += '_opt'
        if val: k += '_bestval'
        k +=  '.npz'
        self._save_optimizer( os.path.join(self._save_model_at_path,k), m )
    if loss_data is not None:
      np.savez(os.path.join( self._save_model_at_path, 'loss.npz'), **loss_data)
    if locals_data is not None:
      np.savez(os.path.join( self._save_model_at_path, 'locals.npz'), **locals_data.__dict__ )

  def _save_model( self, key, model ):
    np.savez( key, model.get_weights() )

  def _save_optimizer( self, key, optimizer ):
    np.savez( key, optimizer.get_weights() )

  def load(self, path, val = False, return_loss = False, return_locals = False, keys = None ):
    if keys is None:
      keys = self._model_dict.keys()
    if not(return_locals or return_loss):
      for k in self._model_dict.keys():
        model = self._model_dict[k]
        optimizer = self._optimizer_dict[k]
        try:
          ko = k
          ko += '_opt'
          if val: ko += '_bestval'
          ko +=  '.npz'
          self._load_optimizer(os.path.join(path,ko), optimizer, model)
        except FileNotFoundError:
          print("Warning: Not recovering optimizer state.")
        if val: 
          k += '_bestval'
        k +=  '.npz'
        self._load_model(os.path.join(path,k), model)
      print("Successfully loaded previous state.")
    if return_loss:
      loss_data_path = os.path.join( path, 'loss.npz' )
      raw_data = dict(**np.load( loss_data_path, allow_pickle=True))
      return self._treat_numpy_data( raw_data )
    if return_locals:
      locals_data_path = os.path.join( path, 'locals.npz' )
      raw_data = dict(**np.load( locals_data_path, allow_pickle=True))
      return self._treat_numpy_data( raw_data )

  def _load_model(self, key, model):
    print("Loading %s weights..." % key)
    try:
      weights = np.load( key, allow_pickle = True )['arr_0']
      model.set_weights( weights )
    except FileNotFoundError:
      model.load_weights( key )
    #model.compile()

  def _load_optimizer(self, key, optimizer, model):
    print("Loading %s optimizer weights..." % key)
    zero_grads = [tf.zeros_like(w) for w in model.trainable_variables]
    saved_vars = [tf.identity(w) for w in model.trainable_variables]
    optimizer.apply_gradients(zip(zero_grads, model.trainable_variables))
    weights = np.load( key, allow_pickle = True )['arr_0']
    optimizer.set_weights( weights )

  def plot_model(self, model_name, *args, **kw):
    if model_name in self._model_dict:
      model = fix_model_layers( self._model_dict[model_name] )
      return tf.keras.utils.plot_model(model, *args, **kw)
    else:
      raise KeyError( "%s is not a valid model key. Available models are: %s" % (model_name, self._model_dict.keys()))

  def _treat_numpy_data( self, raw_loss_data ):
    for k, m in raw_loss_data.items():
      raw_loss_data[k] = m.item()
    return raw_loss_data

  def _plot_train(self, loss_record):
    from IPython.display import display, clear_output
    import matplotlib.pyplot as plt
    if not hasattr(self,"_train_fig"):
      self._train_fig, self._train_ax = plt.subplots()
      first = True
    else:
      first = False
      self._train_ax.cla()
    steps = np.array(loss_record['step'])
    for k, v in loss_record.items():
      if k == "step":
        continue
      v = np.array(v)
      idx = np.where(np.isfinite(v))[0]
      self._train_ax.plot(steps[idx],v[idx],label=k)
    self._train_ax.legend()
    plt.xlabel("#Parameter Updates")
    plt.ylabel("Batch Total Cost")
    plt.tight_layout()
    plt.autoscale(enable=True, axis='x', tight=True)
    clear_output(wait = True)
    display(self._train_fig)

  def _append_loss(self, loss_record, loss_dict, keys = None):
    if keys is None:
      keys = self._lkeys
    for k in keys:
      if k in loss_dict:
        loss_record[k].append(loss_dict[k].numpy() if hasattr(loss_dict[k],"numpy") else loss_dict[k])
      else:
        loss_record[k].append(np.nan)

  def _replace_nans_with_last_report( self, loss_dict, loss_record, keys = None ):
    if keys is None:
      keys = self._lkeys
    for k in keys:
      if k not in loss_dict or not np.isfinite(loss_dict[k]):
        if k not in loss_record: continue
        data = loss_record[k]
        idx = np.where(np.isfinite(data))[0]
        if len(idx):
          loss_dict[k] = data[idx[-1]]

  def _accumulate_loss_dict( self, acc_dict, c_dict):
    for k in c_dict.keys():
      val = c_dict[k].numpy() if hasattr(c_dict[k],"numpy") else c_dict[k]
      if k in acc_dict:
        acc_dict[k] += val
      else:
        acc_dict[k] = val

  def _train_base(self, epoch, step, sample_batch):
    loss_dict = self._train_step(sample_batch)
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
      '; '.join([("%s: %.3f" % (k, v)) for k, v in loss_dict.items() if k is not 'step'])
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

