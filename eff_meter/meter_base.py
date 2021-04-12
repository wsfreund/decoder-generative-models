from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
import datetime


"""
NOTE: When buffering, update_on_parsed_(gen/data) may happen in different order than
otherwise.
"""

class EffMeterBase(ABC):

  def __init__(self, name, data_parser = lambda x: x, **kw):
    self.name = name
    self._initialized = False
    self._deltatime = datetime.timedelta()
    self._step_deltatime = datetime.timedelta()
    self._prev_step_deltatime = datetime.timedelta()
    self._total_running_time = datetime.timedelta()
    self.data_parser = data_parser
    self.data_batch_counter = 0
    self._locked_data_statistics = False

  def update_on_data_batch(self, data, mask = None):
    if self._locked_data_statistics:
      return
    self.start
    self.update_on_parsed_data(self.data_parser(data), mask)
    self.data_batch_counter += 1
    self.stop

  @abstractmethod
  def update_on_parsed_data(self, data, mask):
    pass

  @abstractmethod
  def retrieve(self):
    pass

  def initialize(self, wrapper):
    pass

  @property
  def start(self):
    self._ctime = datetime.datetime.now()

  @property
  def stop(self):
    self._deltatime = datetime.datetime.now() - self._ctime
    self._step_deltatime += self._deltatime
    self._total_running_time += self._deltatime

  def reset(self):
    self._prev_step_deltatime = self._step_deltatime
    self._step_deltatime = datetime.timedelta()
    if not self._locked_data_statistics:
      self.data_batch_counter = 0

  @property
  def print(self):
    print("%s last delta time: %s" % (self.name, self._deltatime))
    print("%s epoch delta time: %s" % (self.name, self._step_deltatime))
    print("%s total computation time: %s" % (self.name, self._total_running_time))

class MultiBatchBuffer(object):
  def __init__(self, max_buffer_size):
    self._max_buffer_size = max_buffer_size
    self._batch_buffer = []
    self._mask_buffer = []
    self._lazy_clear = False

  @property
  def buffer_size(self):
    return len(self._batch_buffer)

  @property
  def is_buffer_full(self):
    return self.buffer_size >= self._max_buffer_size

  def append(self,batch,mask=None):
    if self._lazy_clear:
      self._batch_buffer = []
      self._mask_buffer = []
      self._lazy_clear = False
    if not self._batch_buffer or self._batch_buffer[-1] is not batch:
      self._batch_buffer.append(batch)
      if mask is not None: self._mask_buffer.append(mask)

  def __call__(self):
    return self.retrieve()

  def retrieve(self):
    multi_batch_data = np.concatenate(self._batch_buffer,axis=0) if self._batch_buffer else None
    multi_mask_data = np.concatenate(self._mask_buffer,axis=0) if self._mask_buffer else None
    return multi_batch_data, multi_mask_data

  def clear(self):
    self._batch_buffer = []
    self._mask_buffer = []

  def lazy_clear(self):
    """
    Clear on next call to append
    """
    self._lazy_clear = True

class EffBufferedMeter(EffMeterBase):
  def __init__(self, name, data_parser = lambda x: x, data_buffer = None, max_buffer_size = 16, **kw):
    super().__init__(name = name, data_parser = data_parser, **kw)
    if data_buffer is None:
      data_buffer = MultiBatchBuffer(max_buffer_size)
    self._data_buffer = data_buffer
    # Force update before retrieve
    def force_update(f):
      def local():
        self.update_on_data_batch(data = None, force_update = True)
        return f()
      return local
    self.retrieve = force_update(self.retrieve)

  def update_on_data_batch(self, data, mask = None, force_update = False):
    if self._locked_data_statistics:
      return
    self.start
    if data is not None: 
      self._data_buffer.append(self.data_parser(data), mask)
    if self._data_buffer.is_buffer_full or force_update:
      data, mask = self._data_buffer()
      self.update_on_parsed_data(data, mask)
      self.data_batch_counter += 1
      self._data_buffer.lazy_clear()
    self.stop

class GenerativeEffMeter(EffMeterBase):

  def __init__(self, name, data_parser = lambda x: x, gen_parser = lambda x: x, **kw):
    super().__init__(name = name, data_parser = data_parser, **kw)
    self.gen_parser = gen_parser
    self.gen_batch_counter = 0
    self._locked_gen_statistics = False

  def initialize(self, wrapper):
    self.data_parser = lambda x: wrapper._transform_to_meter_input(x)
    self.gen_parser = lambda x: wrapper._transform_to_meter_input(x)
    super().initialize(wrapper)

  def update_on_gen_batch(self, data, mask = None):
    if self._locked_gen_statistics:
      return
    self.start
    self.update_on_parsed_gen(self.gen_parser(data), mask)
    self.gen_batch_counter += 1
    self.stop

  @abstractmethod
  def update_on_parsed_gen(self, data, mask = None):
    pass

  def reset(self):
    super().reset()
    if not self._locked_gen_statistics:
      self.gen_batch_counter = 0

class GenerativeEffBufferedMeter(EffBufferedMeter, GenerativeEffMeter ):

  def __init__(self, name, data_parser = lambda x: x, data_buffer = None, gen_buffer = None, max_buffer_size = 16, **kw):
    # Force gen update after data update
    def force_update(f):
      def local():
        self.update_on_gen_batch( data = None, force_update = True)
        return f()
      return local
    self.retrieve = force_update(self.retrieve)
    super().__init__(name = name, data_parser = data_parser, data_buffer = data_buffer, max_buffer_size = max_buffer_size, **kw)
    if gen_buffer is None:
      gen_buffer = MultiBatchBuffer(max_buffer_size)
    self._gen_buffer = gen_buffer
    # Force update before retrieve

  def update_on_gen_batch(self, data, mask = None, force_update = False):
    if self._locked_gen_statistics:
      return
    self.start
    if data is not None: 
      self._gen_buffer.append(self.gen_parser(data), mask)
    if self._gen_buffer.is_buffer_full or force_update:
      data, mask = self._gen_buffer()
      self.update_on_parsed_gen(data,mask)
      self.gen_batch_counter += 1
      self._gen_buffer.lazy_clear()
    self.stop
