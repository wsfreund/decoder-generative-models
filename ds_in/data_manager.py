import os
import pandas as pd
import numpy as np
import functools
from abc import ABC, abstractmethod

class TimeseriesMetadata(object):
  seconds_per_hour = 3600
  hours_per_day    = 24
  days_per_week    = 7
  days_per_month   = 30
  days_per_year    = 365.2524
  seconds_per_day   = seconds_per_hour * hours_per_day
  hours_per_week    = hours_per_day    * days_per_week
  seconds_per_week  = seconds_per_hour * hours_per_week
  hours_per_month   = hours_per_day    * days_per_month
  seconds_per_month = seconds_per_hour * hours_per_month
  hours_per_year    = hours_per_day    * days_per_year
  seconds_per_year  = seconds_per_hour * hours_per_year

  @property
  @functools.lru_cache()
  def date_time(self):
    return pd.DatetimeIndex(
        [(self.record_start + self.time_step*i) for i in self.df.index.to_numpy()]
        , freq='infer')

  @property
  @functools.lru_cache()
  def a_day_window_in_samples(self):
    return int(np.round(self.seconds_per_day/self.time_step.total_seconds()))

  @property
  @functools.lru_cache()
  def a_week_window_in_samples(self):
    return int(np.round(self.seconds_per_week/self.time_step.total_seconds()))

  @property
  @functools.lru_cache()
  def a_month_window_in_samples(self):
    return int(np.round(self.seconds_per_month/self.time_step.total_seconds()))

  @property
  @functools.lru_cache()
  def an_year_window_in_samples(self):
    return int(np.round(self.seconds_per_year/self.time_step.total_seconds()))

  @property
  @functools.lru_cache()
  def n_samples_per_year(self):
    return int(np.round(self.seconds_per_year/self.time_step.total_seconds()))

  @property
  @functools.lru_cache()
  def days_in_dataset(self):
    n_samples_h = len(self.df)
    years_per_dataset = (self.date_time[-1]-self.date_time[0]).total_seconds()/self.seconds_per_year
    return years_per_dataset

  @property
  @functools.lru_cache()
  def years_in_dataset(self):
    years_per_dataset = (self.date_time[-1]-self.date_time[0]).total_seconds()/self.seconds_per_year
    return years_per_dataset

class DataManagerBase(ABC):
  _basepaths = []

  def __enter__(self):
    self._read()
    return self

  @abstractmethod
  def __exit__(self, exc_type, exc_value, traceback):
    pass

  @abstractmethod
  def autolocate(self):
    pass

  @abstractmethod
  def _read(self):
    pass

  @property
  def located(self,):
    if not hasattr(self,'_located'):
      self._located = False
    return self._located

  @located.setter
  def located(self, val):
    self._located = val

  def autolocate(self):
    if self.located:
      return
    for p in self._basepaths:
      self._basepath = p
      try:
        return self._get_data_at_path()
      except RuntimeError:
        pass
    # Emergency stop
    return self._get_data_at_path()

class DataFrameManager(DataManagerBase):
  def __init__( self, basefile, shuffle_on_first_read = False ):
    self._basefile = basefile
    self._shuffle_on_first_read = shuffle_on_first_read

  def __exit__(self, exc_type, exc_value, traceback):
    if hasattr(self,'df'):
      del self.df

  def _get_data_at_path( self ):
    formats = [".ft",".csv"]
    found = False
    for f in formats:
      data_path = os.path.join( 
          self._basepath, 
          self._basefile) + f
      if os.path.exists( data_path ):
        found = True
        self.format = f
        break
    if not found:
      raise RuntimeError("Cannot retrieve data at path: %s" % data_path)
    return data_path

  def _read( self ):
    # NOTE Overload this method if a particular dataset has special needs
    data_path = self.autolocate()
    if self.format == ".csv":
      self.df = pd.read_csv( data_path, header = None )
      self.df.columns = map(str,self.df.columns) # ensure that columns are strings
      self.df.to_feather(data_path[:-4]+'.ft')
      if self._shuffle_on_first_read:
        # TODO shuffle on first read for non time series data
        raise NotImplementedError("Shuffle on first read is not implemented")
    elif self.format == ".ft":
      self.df = pd.read_feather( data_path )
    self.data_path = data_path
    return self.df

class PathDataManager(DataManagerBase):
  def __init__( self, relative_path, glob_pattern = '*' ):
    self._relative_path = relative_path 
    self.glob_pattern = glob_pattern

  def __exit__(self, exc_type, exc_value, traceback):
    if hasattr(self,'data_path'):
      del self.data_path

  def _get_data_at_path( self ):
    data_path = os.path.join( 
        self._basepath, 
        self._relative_path)
    if not os.path.exists(data_path):
      raise RuntimeError("Cannot retrieve data at path: %s" % data_path)
    return data_path

  def _read( self ):
    import pathlib
    self.data_path = self.autolocate()
    self.data_path_list = [str(p) for p in pathlib.Path( self.data_path ).glob( self.glob_pattern )]
    return self.data_path_list

