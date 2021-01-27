import os, sys
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tempfile
import itertools
import datetime
import contextlib
from tensorflow.keras import layers
import matplotlib.pyplot as plt

try:
  from misc import *
  from mask_base import MaskModel
  from eff_meter import *
except ImportError:
  from .misc import *
  from .mask_base import MaskModel
  from .eff_meter import *

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
    self._n_performance_measure_steps = retrieve_kw(kw, 'n_performance_measure_steps',     25                                           )
    # Performance functions to be computed on training set
    self._train_perf_meters        = retrieve_kw(kw, 'train_perf_meters',    []                                                         )
    # Performance functions to be computed on validation set
    self._val_perf_meters          = retrieve_kw(kw, 'val_perf_meters',      []                                                         )
    # Must be specified if providing validation dataset, it is the loss key to use for early stopping
    self.early_stopping_key        = retrieve_kw(kw, 'early_stopping_key'                                                               )
    # Maximum number of training epoches (cycles through training dataset)
    self._max_epoches              = retrieve_kw(kw, 'max_epoches',          None                                                       )
    # Maximum number of parameter updates
    self._max_steps                = retrieve_kw(kw, 'max_steps',            None                                                       )
    # Maximum wall time
    self._max_train_wall_time      = retrieve_kw(kw, 'max_wall_time',        None                                                       )
    # Maximum number of fails to improve the validation criterion
    self._max_fail                 = retrieve_kw(kw, 'max_fail',             10000                                                      )
    # Minimum progress on the validation criterion to consider a valid progress
    self._min_progress             = retrieve_kw(kw, 'min_progress',         1e-5                                                       )
    # Whether to log training progress
    self._verbose                  = retrieve_kw(kw, 'verbose',              False                                                      )
    # Specify path to load pre-trained model
    self._load_model_at_path       = retrieve_kw(kw, 'load_model_at_path',   None                                                       )
    # Whether to show online train plot
    self._online_train_plot        = retrieve_kw(kw, 'online_train_plot',    False                                                      )
    # Additional collection of functions to plot during training. Receive the training instance as argument
    self._online_plot_fcns         = retrieve_kw(kw, 'online_plot_fcns',     []                                                         )
    # Interval for logging using updates
    self._print_interval_steps     = retrieve_kw(kw, 'print_interval_steps', 1000                                                       )
    # Interval for logging using wall time
    self._print_interval_wall_time = retrieve_kw(kw, 'print_interval_wall_time', datetime.timedelta( seconds = 15 )                     )
    # Interval for logging using epoches
    self._print_interval_epoches   = retrieve_kw(kw, 'print_interval_epoches', 5                                                        )
    # Path to log tensorboard data
    self._tensorboard_log_path     = retrieve_kw(kw, 'tensorboard_log_path', 'tensorboard_logs'                                         )
    tensorboard_key = os.path.expandvars("$TENSORBOARD_LOGGING_KEY")
    user = os.path.expandvars("$USER")
    if tensorboard_key != "$TENSORBOARD_LOGGING_KEY":
      self._tensorboard_log_path += "_" + tensorboard_key
    elif user != "$USER":
      self._tensorboard_log_path += "_" + user
    else:
      pass
    # String specifying model label
    self._model_name               = retrieve_kw(kw, 'model_name', self.__class__.__name__                                              )
    # String specifying model first initialization time
    self._init_time                = retrieve_kw(kw, 'init_time', datetime.datetime.now().strftime("%Y%m%d-%H%M%S")                     )
    # Steps interval for saving current progress (model weights and training log)
    self._save_interval_steps      = retrieve_kw(kw, 'save_interval_steps',   None                                                      )
    # Wall time iterval for saving current progress (model weights and training log)
    self._save_interval_wall_time  = retrieve_kw(kw, 'save_interval_wall_time', datetime.timedelta( minutes = 5 )                       )
    # Epoches interval for saving current progress (model weights and training log)
    self._save_interval_epoches    = retrieve_kw(kw, 'save_interval_epoches',  None                                                     )
    # Use log-sampling periods of loss functions
    self._use_log_history          = retrieve_kw(kw, 'use_log_history',      True                                                       )
    # Number of history samples when using log-sampled history
    self._history_max_batch_samples = retrieve_kw(kw, 'log_n_linear_history_samples',    50                                             )
    # Log-sampling period. To sample more, use lower values, i.e. 0.005. To sample less, higher values, i.e. 0.05
    self._log_sampling_period      = retrieve_kw(kw, 'log_sampling_period',  0.01                                                       )
    # File path to be used when saving/loading
    self._save_model_at_path       = retrieve_kw(kw, 'save_model_at_path',   "trained_model"                                            )
    # Whether to apply gradient clipping
    self._use_grad_clipping        = tf.constant( retrieve_kw(kw, 'use_grad_clipping', False  ) )
    # Gradient clipping function
    self._grad_clipping_fcn        = retrieve_kw(kw, 'grad_clipping_fcn', lambda x: tf.clip_by_norm( x, 2.0 )  )
    # Maximum number of samples to use when evaluating performance
    ## Setup
    # lkeys and val_lkeys are used to select which losses are to be recorded
    self._surrogate_lkeys  = {"step",}
    self._train_perf_lkeys = {"step",} | set(map(lambda m: m.name, self._train_perf_meters))
    self._val_perf_lkeys   = {"step",} | set(map(lambda m: m.name, self._val_perf_meters))
    self._model_dict = {}
    self._optimizer_dict = {}
    # Define summary writers:
    if self._tensorboard_log_path:  
      self._surrogate_summary_writer  = self._create_writer( "surrogate" )
      self._train_perf_summary_writer = self._create_writer( "train_perf" )
      self._val_perf_summary_writer   = self._create_writer( "val_perf" )
    else:
      self._surrogate_summary_writer  = contextlib.suppress()
      self._train_perf_summary_writer = contextlib.suppress()
      self._val_perf_summary_writer   = contextlib.suppress()
    ## build models
    if not hasattr(self,'_required_models'):
      raise RuntimeError("Class '%s' does not define any required model." % self.__class__.__name__)
    self._model_io_keys = set()
    self._model_dict.update( self.build_models() )
    self._decorate_models()
    ## Sanity checks
    if self._max_train_wall_time is not None:
      assert isinstance(self._max_train_wall_time, datetime.timedelta)
    if self._print_interval_wall_time is not None:
      assert isinstance(self._print_interval_wall_time, datetime.timedelta)
    for meter in self._train_perf_meters:
      assert isinstance(meter,EffMeterBase)
    for meter in self._val_perf_meters:
      assert isinstance(meter,EffMeterBase)

  def train(self, fine_tuning = False):
    self._check_required_models()
    start_train_wall_time = datetime.datetime.now()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    n_gpus = len(gpus)
    if self._verbose: print('This machine has %i GPUs.' % n_gpus)

    # containers for losses
    lc = Container()
    if self._load_model_at_path is not None:
      # Load model
      self.load( self._load_model_at_path )
      # Load loss record
      loss_data = self.load( self._load_model_at_path, return_loss = True)
      try:
        lc.surrogate_loss_record  = loss_data["surrogate_loss_record"]
      except KeyError:
        lc.surrogate_loss_record  = loss_data["train_record"]
      try:
        lc.train_perf_record  = loss_data["train_perf_record"]
      except KeyError:
        lc.train_perf_record = {k : [] for k in self._train_perf_lkeys}
      try:
        lc.val_perf_record  = loss_data["val_perf_record"]
      except KeyError:
        lc.val_perf_record  = loss_data["val_record"]
      # Load local_keys
      lc.__dict__.update( self.load( self._load_model_at_path, return_locals = True) )
      # Save a model copy with previous best validation so that if this
      # training is useless, we can recover previous best model
      self.save( overwrite = True, val = True )
    else:
      lc.surrogate_loss_record = {k : [] for k in self._surrogate_lkeys}
      lc.train_perf_record = {k : [] for k in self._train_perf_lkeys}
      lc.val_perf_record = {k : [] for k in self._val_perf_lkeys}
      lc.best_epoch = lc.best_step = lc.last_progress_step = 0; lc.best_val_reco = np.finfo( dtype = np.float32 ).max
      lc.p_sample_period = -np.inf; lc._history_cur_batch_samples = 0;
      lc.step = lc.epoch = 0
    last_print_cycle = -1; last_save_cycle = 0
    skipFinalPerfEval = is_new_print_cycle = False;
    exc_type = exc_val =  None
    # When fine tuning, we need to reset the validation dataset statistics
    if fine_tuning:
      lc.best_val_reco = np.finfo( dtype = np.float32 ).max

    train_perf_dict = {}
    val_perf_dict = {}
    first_step = True
    total_performance_measure_time = datetime.timedelta()
    n_measurements = 0

    try:
      while (lc.epoch < self._max_epoches if self._max_epoches else True):
        alreadyPrintedEpoch = alreadySavedEpoch = False
        for sample_batch in self.data_sampler.training_sampler:
          evaluatedPerf = False
          # TODO To measure performance on purely initialize sample, simply run
          # below without running self._train_base and without incrementing
          # lc.step
          if self.sample_parser_fcn is not None:
            sample_batch = self.sample_parser_fcn(sample_batch)
          #print("Running first train step")
          surrogate_loss_dict = self._train_base(lc.epoch, lc.step, sample_batch)
          #print("Finished computing and updating one train step")
          lc.step += 1
          start_performance_measure = datetime.datetime.now()
          surrogate_loss_dict = self._parse_surrogate_loss( surrogate_loss_dict )
          surrogate_loss_dict['step'] = lc.step
          # Keep track of training record:
          # TODO This should be integrated in the meters, i.e. compute loss only if passing constraint below
          c_sample_period = np.log10( lc.step ) // self._log_sampling_period
          if c_sample_period  > lc.p_sample_period:
            lc.p_sample_period = c_sample_period
            lc._history_cur_batch_samples = 0
          if lc._history_cur_batch_samples < self._history_max_batch_samples:
            # NOTE handle_new_loss_step keeps track of what is plot/logged
            # during training
            # TODO This should be integrated with the meter framework
            #print("Keeping track of surrogate loss")
            with (self._surrogate_summary_writer.as_default(step = lc.step) if hasattr(self._train_perf_summary_writer, 'as_default') 
                else self._surrogate_summary_writer ) as writer:
              self._handle_new_loss_step(lc.surrogate_loss_record, surrogate_loss_dict, write_to_summary = writer is not None )
          # Compute efficiency
          if ( not(lc.step % self._n_performance_measure_steps) or lc.step == 1) and self._has_performance_measure_fcn:
            n_measurements += 1
            #print("Computing train dataset performance")
            #if lc._history_cur_batch_samples < self._history_max_batch_samples:
            with (self._train_perf_summary_writer.as_default(step = lc.step) if hasattr(self._train_perf_summary_writer, 'as_default') 
                else self._train_perf_summary_writer ) as writer:
              train_perf_dict = self.performance_measure_fcn(
                  sampler_ds = self.data_sampler.evaluation_sampler_from_train_ds,
                  meters = self._train_perf_meters, lc = lc)
              train_perf_dict['step'] = lc.step
              self._handle_new_loss_step(lc.train_perf_record, train_perf_dict, keys = self._train_perf_lkeys, write_to_summary = writer is not None )
            # Compute performance for validation dataset (when available)
            if self.data_sampler.has_val_ds:
              #print("Computing val dataset performance")
              with (self._val_perf_summary_writer.as_default(step = lc.step) if hasattr(self._val_perf_summary_writer, 'as_default') 
                  else self._val_perf_summary_writer ) as writer:
                val_perf_dict = self.performance_measure_fcn(
                    sampler_ds = self.data_sampler.evaluation_sampler_from_val_ds,
                    meters = self._val_perf_meters, lc = lc )
                val_perf_dict['step'] = lc.step
                evaluatedPerf = True
                # TODO Reminder handle new loss keeps track of what is plot/logged
                # during training. Should be integrated with meter framework
                #if lc._history_cur_batch_samples < self._history_max_batch_samples:
                self._handle_new_loss_step(lc.val_perf_record, val_perf_dict, keys = self._val_perf_lkeys, write_to_summary = writer is not None )
              # Early stopping algo: Keep track of best model so far
              #print("Computing early stopping")
              if val_perf_dict[self.early_stopping_key] < lc.best_val_reco:
                if lc.best_val_reco - val_perf_dict[self.early_stopping_key] > self._min_progress:
                  lc.last_progress_step = lc.step
                lc.best_val_reco = val_perf_dict[self.early_stopping_key]
                lc.best_step = lc.step; lc.best_epoch = lc.epoch 
                self.save( overwrite = True, val = True )
              # Check whether to break due to 
              if ( lc.step - lc.last_progress_step ) >= self._max_fail:
                raise BreakDueToMaxFail()
          # End of efficiency computation
          # Performed one model update step
          # Compute training time
          train_time = datetime.datetime.now() - start_train_wall_time
          print_cycle = int( train_time / self._print_interval_wall_time ) if self._print_interval_wall_time is not None else 0
          is_new_print_cycle = print_cycle > last_print_cycle
          # Compute performance measurement time:
          stop_performance_measure = datetime.datetime.now()
          this_step_performance_measure_time = stop_performance_measure - start_performance_measure
          if evaluatedPerf:
            total_performance_measure_time += this_step_performance_measure_time
            last_performance_measure_time = this_step_performance_measure_time
            if first_step:
              first_step_measure_time = this_step_performance_measure_time
              first_step = False
          # Print/plot loss
          if ((self._verbose or self._online_train_plot) and 
                (
                  (not(lc.step % self._print_interval_steps) if self._print_interval_steps is not None else False) 
                  or ((not(lc.epoch % self._print_interval_epoches)  if self._print_interval_epoches is not None else False) and not(alreadyPrintedEpoch) )
                  or is_new_print_cycle
                )
              ):
            #print("Proceeding to printing")
            last_improvement = { 'best_val_reco' : lc.best_val_reco
                               , 'best_step' : lc.best_step
                               , 'last_progress_step' : lc.last_progress_step } if val_perf_dict else {}
            if self._online_train_plot:
              try:
                from google.colab import output
                output.clear()
              except ImportError:
                from IPython.display import clear_output
                clear_output(wait = True)
              plt.close('all')
              self._plot_surrogate_progress( lc.surrogate_loss_record )
              self._plot_performance_progress( lc.train_perf_record, lc.val_perf_record )
              for fcn in self._online_plot_fcns:
                fcn()
            if self._verbose: 
              self._replace_nans_with_last_report( surrogate_loss_dict, lc.surrogate_loss_record )
              self._print_progress( lc.epoch, lc.step
                                  , train_time, total_performance_measure_time, last_performance_measure_time, first_step_measure_time
                                  , n_measurements
                                  , surrogate_loss_dict, train_perf_dict, val_perf_dict, last_improvement )
            if not(lc.epoch % self._print_interval_epoches) if self._print_interval_epoches is not None else False:
              alreadyPrintedEpoch = True
            if is_new_print_cycle:
              last_print_cycle = print_cycle
              is_new_print_cycle = False
          # Finished printing
          # Check whether we have finished training
          if self._max_steps is not None and (lc.step > self._max_steps):
            raise BreakDueToUpdates()
          if self._max_train_wall_time is not None and (train_time > self._max_train_wall_time):
            raise BreakDueToWallTime()
          # No halt requested. Increament counters
          if lc._history_cur_batch_samples < self._history_max_batch_samples:
            lc._history_cur_batch_samples += 1
          save_cycle = int( train_time / self._save_interval_wall_time ) if self._save_interval_wall_time is not None else 0
          is_new_save_cycle = save_cycle > last_save_cycle
          # Save progress. Note that this save is not due to early stopping,
          # but rather to allow recovering current weights regardless of
          # training status
          if (
              (not(lc.step % self._save_interval_steps) if self._save_interval_steps is not None else False) 
              or ((not(lc.epoch % self._save_interval_epoches)  if self._save_interval_epoches is not None else False) and not(alreadySavedEpoch) )
              or is_new_save_cycle
             ):
            #print("Saving progress")
            loss_data = { 'surrogate_loss_record' : lc.surrogate_loss_record
                        , 'train_perf_record' : lc.train_perf_record
                        , 'val_perf_record' : lc.val_perf_record }
            self.save( overwrite = True
                , loss_data = loss_data
                , locals_data = lc )
            if not(lc.epoch % self._save_interval_epoches) if self._save_interval_epoches is not None else False:
              alreadySavedEpoch = True
            if is_new_save_cycle:
              last_save_cycle = save_cycle
              is_new_save_cycle = False
        lc.epoch += 1
        # Performed a full pass through training dataset
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
          print('Recovering Best Validation Performance @ (Epoch %i, Step %i).' % (lc.best_epoch, lc.best_step,))
          print('Reco_loss: %.3f.' % lc.best_val_reco)
          self.load( self._save_model_at_path, val = True )
          skipFinalPerfEval = True
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
      if self.data_sampler.has_val_ds and self._has_performance_measure_fcn:
        if not skipFinalPerfEval:
          if not evaluatedPerf:
            with (self._train_perf_summary_writer.as_default(step = lc.step) if hasattr(self._train_perf_summary_writer, 'as_default') 
                else self._train_perf_summary_writer) as writer:
              train_perf_dict = self.performance_measure_fcn( 
                  sampler_ds = self.data_sampler.evaluation_sampler_from_train_ds,
                  meters = self._train_perf_meters, lc = lc
              )
              train_perf_dict['step'] = lc.step
              self._handle_new_loss_step(lc.train_perf_record, train_perf_dict, keys = self._train_perf_lkeys, write_to_summary = writer is not None)
            with (self._val_perf_summary_writer.as_default(step = lc.step) if hasattr(self._val_perf_summary_writer, 'as_default') 
                else self._val_perf_summary_writer) as writer:
              val_perf_dict = self.performance_measure_fcn( 
                  sampler_ds = self.data_sampler.evaluation_sampler_from_val_ds,
                  meters = self._val_perf_meters, lc = lc
              )
              val_perf_dict['step'] = lc.step
              self._handle_new_loss_step(lc.val_perf_record, val_perf_dict, keys = self._val_perf_lkeys, write_to_summary = writer is not None)
          if lc.step == lc.best_step or val_perf_dict[self.early_stopping_key] < lc.best_val_reco:
            lc.best_val_reco = val_perf_dict[self.early_stopping_key]
            lc.best_step = lc.step; lc.best_epoch = lc.epoch 
          else:
            print('Validation Performance @ (Epoch %i, Step %i): %f.' % (lc.epoch, lc.step, val_perf_dict[self.early_stopping_key]))
            print('Recovering Best Validation Performance @ (Epoch %i, Step %i).' % (lc.best_epoch, lc.best_step,))
            print('Reco_loss: %.3f.' % (lc.best_val_reco))
            self.load(  self._save_model_at_path, val = True )
    self.save( overwrite = True, locals_data = lc )
    # Compute final performance:
    final_performance = {}
    if self._has_performance_measure_fcn:
      final_performance['train'] = self.performance_measure_fcn(
          sampler_ds = self.data_sampler.evaluation_sampler_from_train_ds,
          meters = self._train_perf_meters,  lc = lc )
      if self.data_sampler.has_val_ds:
        final_performance['val'] = self.performance_measure_fcn(
            sampler_ds = self.data_sampler.evaluation_sampler_from_val_ds,
            meters = self._val_perf_meters, lc = lc )
        final_performance['val']['best_step'] = lc.best_step
        final_performance['val']['best_epoch'] = lc.best_epoch
      else:
        final_performance['val'] = dict()
    loss_data = { 'surrogate_loss_record' : lc.surrogate_loss_record
                , 'train_perf_record' : lc.train_perf_record
                , 'val_perf_record' : lc.val_perf_record
                , 'final_performance' : final_performance }
    self.save( save_models_and_optimizers = False
             , loss_data = loss_data)
    return loss_data

  def build_models(self):
    raise NotImplementedError("Overload this method returning a dict with the models to be used.")

  def loss_per_dataset(self, fcn = None):
    if fcn is None:
      if self._has_performance_measure_fcn: 
        fcn = self.performance_measure_fcn
      else:
        raise ValueError("Performance measure function must be specified")
    return { 'train': fcn(self.data_sampler.new_sampler_from_train_ds( **self._eval_ds_kwargs ) )
           , 'val':   fcn(self.data_sampler.new_sampler_from_val_ds( **self._eval_ds_kwargs ) )
           , 'test':  fcn(self.data_sampler.new_sampler_from_test_ds( **self._eval_ds_kwargs ) ) }

  def save(self, overwrite = False, save_models_and_optimizers = True, val = False, loss_data = None, locals_data = None ):
    if not self._model_io_keys:
      raise ValueError("Empty model io keys for class %s" % self.__class__.__name__)
    # Create folder if it does not exist
    if not os.path.exists(self._save_model_at_path):
      os.makedirs(self._save_model_at_path)
    if save_models_and_optimizers:
      for k in self._model_io_keys:
        m = self._model_dict[k]
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
    np.savez( key, np.array(model.get_weights(), dtype=np.object) )

  def _save_optimizer( self, key, optimizer ):
    np.savez( key, np.array(optimizer.get_weights(), dtype=np.object) )

  def load(self, path, val = False, return_loss = False, return_locals = False ):
    if not self._model_io_keys:
      raise ValueError("Empty model io keys for class %s" % self.__class__.__name__)
    if not(return_locals or return_loss):
      for ko, optimizer in self._optimizer_dict.items():
        model = self._model_dict[ko]
        try:
          ko += '_opt'
          if val: ko += '_bestval'
          ko +=  '.npz'
          if optimizer is not None:
            self._load_optimizer(os.path.join(path,ko), optimizer, model)
        except FileNotFoundError:
          print("Warning: Could not recover %s optimizer state." % ko)
      for k in self._model_io_keys:
        model = self._model_dict[k]
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

  def performance_measure_fcn(self, sampler_ds, meters_dict, lc):
    raise RuntimeError("Performance measure function is not implemented")

  def _create_writer( self, output_place ): 
    return tf.summary.create_file_writer( os.path.join( self._tensorboard_log_path, self._model_name, self._init_time, output_place ) )

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

  def _check_required_models(self):
    for model_key in self._required_models:
      if not model_key in self._model_dict:
        raise RuntimeError("Model %s was not provided." % model_key )

  def _decorate_models(self):
    for model_key, model in self._model_dict.items():
      if not hasattr(self,model_key):
        model.compile()
        setattr(self,model_key,model)
      else:
        raise RuntimeError("Duplicated model %s." % model_key )

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

  def _plot_surrogate_progress(self, surrogate_loss_record):
    from IPython.display import display
    if not hasattr(self,"_surrogate_fig"):
      self._surrogate_fig, self._surrogate_ax = plt.subplots()
    else:
      self._surrogate_ax.cla()
    steps = np.array(surrogate_loss_record['step'])
    for k, v in surrogate_loss_record.items():
      if k == "step":
        continue
      v = np.array(v)
      idx = np.where(np.isfinite(v))[0]
      self._surrogate_ax.plot(steps[idx],v[idx],label=k)
    self._surrogate_ax.autoscale(enable=True, axis='x', tight=True)
    self._surrogate_ax.set_xlabel("#Parameter Updates")
    self._surrogate_ax.set_ylabel("Surrogate Loss")
    self._surrogate_ax.legend()
    display(self._surrogate_fig)

  def _plot_performance_progress(self, train_perf_record, val_perf_record):
    from IPython.display import display
    if not hasattr(self,"_perf_fig"):
      self._all_perf_keys = set(val_perf_record.keys()) | set(train_perf_record.keys())
      for k in ("step", "critic_gen", "critic_data"): # XXX
        if k in self._all_perf_keys: 
          self._all_perf_keys.remove(k)
          if k.startswith("critic"):
            self._all_perf_keys |= {"critic"}
      n_keys = len(self._all_perf_keys)
      self._perf_fig, self._all_perf_ax = plt.subplots(n_keys,1,sharex=True, gridspec_kw={'hspace': 0})
      self._perf_final_ax = self._perf_fig.add_subplot(111, frameon=False)
      self._perf_final_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
      self._perf_final_ax.grid(False)
      self._perf_final_ax.set_xlabel("#Parameter Updates")
      self._perf_final_ax.set_ylabel("Performance/Loss")
      if n_keys == 1:
        self._all_perf_ax = [self._all_perf_ax]
    else:
      for ax in self._all_perf_ax: ax.cla()
    steps = np.array(val_perf_record['step'])
    def add_plot(ax, steps, record, key, label):
      v = np.array(record[key])
      idx = np.where(np.isfinite(v))[0]
      ax.plot(steps[idx], v[idx], label=label)
    if len(steps):
      for ax, k in zip(self._all_perf_ax, self._all_perf_keys):
        if k in train_perf_record or k + "_data" in train_perf_record:
          if k == "critic": # XXX
            add_plot(ax, steps, train_perf_record, k + "_data", k + "(data,train)")
            add_plot(ax, steps, train_perf_record, k + "_gen", k + "(gen,train)")
          else:
            add_plot(ax, steps, train_perf_record, k, k + " (train)")
        if k in val_perf_record or k + "_data" in val_perf_record:
          if k == "critic": # XXX
            add_plot(ax, steps, val_perf_record, k + "_data", k + "(data,val)")
            add_plot(ax, steps, val_perf_record, k + "_gen", k + "(gen,val)")
          else:
            add_plot(ax, steps, val_perf_record, k, k + " (val)")
        ax.legend(prop={'size': 6})
        ax.autoscale(enable=True, axis='x', tight=True)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=6)
      plt.tight_layout()
      display(self._perf_fig)

  def _handle_new_loss_step(self, loss_record, loss_dict, keys = None, write_to_summary = False):
    if keys is None:
      keys = self._surrogate_lkeys
    for k in keys:
      if k in loss_dict:
        eff = loss_dict[k]
        # TODO if isinstance(eff, Efficiency)
        val = loss_dict[k].numpy() if hasattr(loss_dict[k],"numpy") else loss_dict[k]
        if hasattr(val,"shape") and val.shape: val = val[0]
      else:
        val = np.nan
      loss_record[k].append(val)
      if write_to_summary and k not in ("step", "critic_fake"): # XXX
        tf.summary.scalar( k, val, step = loss_dict['step'] )

  def _replace_nans_with_last_report( self, loss_dict, loss_record, keys = None ):
    if keys is None:
      keys = self._surrogate_lkeys
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
    surrogate_loss_dict = self._train_step(sample_batch)
    return surrogate_loss_dict

  def _parse_surrogate_loss(self, train_loss):
    for k, v in train_loss.items():
      if tf.math.logical_not(tf.math.is_finite(v)):
        raise BreakDueToNonFinite(k)
    return train_loss

  @property
  def _has_performance_measure_fcn(self): 
    return hasattr(self,'performance_measure_fcn')

  def _print_progress( self, epoch, step
                     , train_time, total_performance_measure_time, last_performance_measure_time, first_step_measure_time
                     , n_measurements
                     , surrogate_loss_dict, train_perf_dict, val_perf_dict
                     , last_improvement ):
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
    print(('>>Epoch: %i. Steps: %i. Time: %s. Training %s complete.\n::Surrogate: ' % (epoch, step, train_time, perc_str)) + 
      '; '.join([("%s: %.3f" % (k, v)) for k, v in surrogate_loss_dict.items() if k is not 'step']) + "."
    )
    if train_perf_dict or val_perf_dict:
      lost_steps = step * ( total_performance_measure_time / ( train_time - total_performance_measure_time ) )
      lost_frac = ( step + lost_steps ) / step  - 1.
      print('::Performance @ step %i:' % train_perf_dict['step'] )
      print('...Runtime overhead: last: %s; avg: %s (n=%d); total: %s (eff:%s%%|lost:%4.0f|incr:%s%%); first: %s.' % 
            ( last_performance_measure_time
            , ( total_performance_measure_time - first_step_measure_time ) / ( n_measurements - 1 ) if ( n_measurements - 1 ) > 0 else '---'
            , n_measurements
            , total_performance_measure_time 
            , np.around(100*(1.-total_performance_measure_time/train_time), decimals=1) 
            , lost_steps
            , np.around(100*lost_frac, decimals=1)
            , first_step_measure_time )
      )
      #if not hasattr(self,"_last_shown_perf"):
      #  self._last_shown_perf = -1
      #if val_perf_dict['step'] != self._last_shown_perf:
      # self._last_shown_perf = val_perf_dict['step']
      if train_perf_dict:
        print('...Train: ' + self.early_stopping_key + "_train" + (': %.3f; ' % (train_perf_dict[self.early_stopping_key])) +
          '; '.join([("%s: %.3f" % (k + "_train", v)) for k, v in train_perf_dict.items() if k not in ('step', self.early_stopping_key)]) + '.'
        )
      if val_perf_dict:
        print('...Validation: ' +
          self.early_stopping_key + "_val" + (': %.3f; ' % (val_perf_dict[self.early_stopping_key])) +
          '; '.join([("%s: %.3f" % (k + "_val", v)) for k, v in val_perf_dict.items() if k not in ('step', self.early_stopping_key)]) + '.'
        )
      if last_improvement:
        delta_fail = (step - last_improvement['last_progress_step']) if (step - last_improvement['last_progress_step']) > self._n_performance_measure_steps else 0
        print( ( '...Best Val: %.3f (step=%d) =>'  % 
            ( last_improvement['best_val_reco']
            , last_improvement['best_step'] ) )
            + ( ( ' Fails = [%d/%d]' % ( 
                "(<min prog)" if ( last_improvement['best_step'] != last_improvement['last_progress_step']) else ""
              , delta_fail
              , self._max_fail
              ) ) 
            if delta_fail 
            else ' Improved.' )
        )
