from abc import ABC, abstractmethod
import tensorflow as tf
import datetime

class EffMeterBase(ABC):

  def __init__(self, name, data_parser = lambda x: x):
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

class GenerativeEffMeter(EffMeterBase):

  def __init__(self, name, data_parser = lambda x: x, gen_parser = lambda x: x):
    super().__init__(name, data_parser)
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

