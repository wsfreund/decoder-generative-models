"""
Adaptation of ResNet models for Keras to 1D inputs.
"""
from tensorflow.python.keras import backend
from tensorflow.keras import layers

def ResNet(stack_fn, seq_input, preact, use_bias, model_name='resnet'):
  """Instantiates the ResNet, ResNetV2, and ResNeXt architecture for 1D inputs.

  Reference:
  - [Deep Residual Learning for Image Recognition](
      https://arxiv.org/abs/1512.03385) (CVPR 2015)

  Arguments:
    stack_fn: a function that returns output tensor for the
      stacked residual blocks.
    seq_input: the tensor to be processed.
    preact: whether to use pre-activation or not
      (True for ResNetV2, False for ResNet and ResNeXt).
    use_bias: whether to use biases for convolutional layers or not
      (True for ResNet and ResNetV2, False for ResNeXt).
    model_name: string, model name.
  Returns:
    A `keras.Model` instance.

  Raises:
    ValueError: in case of invalid argument for `weights`,
      or invalid input shape.
    ValueError: if `classifier_activation` is not `softmax` or `None` when
      using a pretrained top layer.
  """
  if kwargs:
    raise ValueError('Unknown argument(s): %s' % (kwargs,))

  bn_axis = 2 if backend.image_data_format() == 'channels_last' else 1

  # This is the top
  #x = layers.ZeroPadding1D(padding=((3,), (3,)), name='conv1_pad')(seq_input)
  #x = layers.Conv1D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)

  #if not preact:
  #  x = layers.BatchNormalization( axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
  #  x = layers.Activation('relu', name='conv1_relu')(x)

  #x = layers.ZeroPadding1D(padding=((1,), (1,)), name='pool1_pad')(x)
  #x = layers.MaxPooling1D(3, strides=2, name='pool1_pool')(x)

  x = stack_fn(x)

  if preact:
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='post_bn')(x)
    x = layers.Activation('relu', name='post_relu')(x)

  ## Ensure that the model takes into account
  ## any potential predecessors of `input_tensor`.
  #from tensorflow.python.keras.utils import layer_utils
  #if input_tensor is not None:
  #  inputs = layer_utils.get_source_inputs(input_tensor)
  #else:
  #  inputs = img_input

  ## Create model.
  #from tensorflow.python.keras.engine import training
  #model = training.Model(inputs, x, name=model_name)

  return x


def block1(x, filters, kernel_size=3, stride=1, kernel_initializer='glorot_uniform', layer_kwargs={}, bottleneck=True, conv_shortcut=True, name=None):
  """A residual block.
  Arguments:
    x: input tensor.
    filters: integer, filters of the bottleneck layer.
    kernel_size: default 3, kernel size of the bottleneck layer.
    stride: default 1, stride of the first layer.
    conv_shortcut: default True, use convolution shortcut if True,
        otherwise identity shortcut.
    name: string, block label.

  Returns:
    Output tensor for the residual block.
  """
  bn_axis = 2 if backend.image_data_format() == 'channels_last' else 1

  if conv_shortcut:
    if bottleneck:
      shortcut = layers.Conv1D(4 * filters, 1, strides=stride, kernel_initializer=kernel_initializer, name=name + '_0_conv', **layer_kwargs)(x)
    else:
      shortcut = layers.Conv1D(filters, 1, strides=stride, kernel_initializer=kernel_initializer, name=name + '_0_conv', **layer_kwargs)(x)
    shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
  else:
    shortcut = x

  x = layers.Conv1D(filters, 1, strides=stride, kernel_initializer=kernel_initializer, name=name + '_1_conv', **layer_kwargs)(x)
  x = layers.BatchNormalization( axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
  x = layers.Activation('relu', name=name + '_1_relu')(x)

  x = layers.Conv1D( filters, kernel_size, padding='SAME', kernel_initializer=kernel_initializer, name=name + '_2_conv', **layer_kwargs)(x)
  x = layers.BatchNormalization( axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn', gamma_initializer='zeros' if not bottleneck else 'ones' )(x)

  if bottleneck:
    x = layers.Activation('relu', name=name + '_2_relu')(x)
    x = layers.Conv1D(4 * filters, 1, kernel_initializer=kernel_initializer, name=name + '_3_conv', **layer_kwargs)(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn', gamma_initializer='zeros')(x)

  x = layers.Add(name=name + '_add')([shortcut, x])
  x = layers.Activation('relu', name=name + '_out')(x)
  return x


def stack1(x, filters, blocks, kernel_size=3, stride1=2, kernel_initializer='glorot_uniform', layer_kwargs={}, name=None):
  """A set of stacked residual blocks.

  Arguments:
    x: input tensor.
    filters: integer, filters of the bottleneck layer in a block.
    blocks: integer, blocks in the stacked blocks.
    stride1: default 2, stride of the first layer in the first block.
    name: string, stack label.

  Returns:
    Output tensor for the stacked blocks.
  """
  x = block1(x, filters, kernel_size=kernel_size, stride=stride1, kernel_initializer=kernel_initializer, name=name + '_block1', layer_kwargs=layer_kwargs)
  for i in range(2, blocks + 1):
    x = block1(x, filters, kernel_size=kernel_size, conv_shortcut=False, kernel_initializer=kernel_initializer, name=name + '_block' + str(i), layer_kwargs=layer_kwargs)
  return x


def block2(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
  """A residual block.

  Arguments:
      x: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default False, use convolution shortcut if True,
        otherwise identity shortcut.
      name: string, block label.

  Returns:
    Output tensor for the residual block.
  """
  bn_axis = 2 if backend.image_data_format() == 'channels_last' else 1

  preact = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_preact_bn')(x)
  preact = layers.Activation('relu', name=name + '_preact_relu')(preact)

  if conv_shortcut:
    shortcut = layers.Conv1D(
        4 * filters, 1, strides=stride, name=name + '_0_conv')(preact)
  else:
    shortcut = layers.MaxPooling1D(1, strides=stride)(x) if stride > 1 else x

  x = layers.Conv1D(
      filters, 1, strides=1, use_bias=False, name=name + '_1_conv')(preact)
  x = layers.BatchNormalization( axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
  x = layers.Activation('relu', name=name + '_1_relu')(x)

  x = layers.ZeroPadding1D(padding=((1,), (1,)), name=name + '_2_pad')(x)
  x = layers.Conv1D( filters, kernel_size, strides=stride, use_bias=False, name=name + '_2_conv')(x)
  x = layers.BatchNormalization( axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
  x = layers.Activation('relu', name=name + '_2_relu')(x)

  x = layers.Conv1D(4 * filters, 1, name=name + '_3_conv')(x)
  x = layers.Add(name=name + '_out')([shortcut, x])
  return x


def stack2(x, filters, blocks, stride1=2, name=None):
  """A set of stacked residual blocks.

  Arguments:
      x: input tensor.
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the first layer in the first block.
      name: string, stack label.

  Returns:
      Output tensor for the stacked blocks.
  """
  x = block2(x, filters, conv_shortcut=True, name=name + '_block1')
  for i in range(2, blocks):
    x = block2(x, filters, name=name + '_block' + str(i))
  x = block2(x, filters, stride=stride1, name=name + '_block' + str(blocks))
  return x


def block3(x, filters, kernel_size=3, stride=1, groups=32, conv_shortcut=True, name=None):
  """A residual block.

  Arguments:
    x: input tensor.
    filters: integer, filters of the bottleneck layer.
    kernel_size: default 3, kernel size of the bottleneck layer.
    stride: default 1, stride of the first layer.
    groups: default 32, group size for grouped convolution.
    conv_shortcut: default True, use convolution shortcut if True,
        otherwise identity shortcut.
    name: string, block label.

  Returns:
    Output tensor for the residual block.
  """
  bn_axis = 2 if backend.image_data_format() == 'channels_last' else 1

  if conv_shortcut:
    shortcut = layers.Conv1D( (64 // groups) * filters, 1, strides=stride, use_bias=False, name=name + '_0_conv')(x)
    shortcut = layers.BatchNormalization( axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
  else:
    shortcut = x

  x = layers.Conv1D(filters, 1, use_bias=False, name=name + '_1_conv')(x)
  x = layers.BatchNormalization( axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
  x = layers.Activation('relu', name=name + '_1_relu')(x)

  c = filters // groups
  x = layers.ZeroPadding1D(padding=((1,), (1,)), name=name + '_2_pad')(x)
  x = layers.DepthwiseConv2D( kernel_size, strides=stride, depth_multiplier=c, use_bias=False, name=name + '_2_conv')(x) # TODO
  x_shape = backend.int_shape(x)[1:-1]
  x = layers.Reshape(x_shape + (groups, c, c))(x)
  x = layers.Lambda( lambda x: sum(x[:, :, :, :, i] for i in range(c)), name=name + '_2_reduce')(x)
  x = layers.Reshape(x_shape + (filters,))(x)
  x = layers.BatchNormalization( axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
  x = layers.Activation('relu', name=name + '_2_relu')(x)

  x = layers.Conv1D( (64 // groups) * filters, 1, use_bias=False, name=name + '_3_conv')(x)
  x = layers.BatchNormalization( axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

  x = layers.Add(name=name + '_add')([shortcut, x])
  x = layers.Activation('relu', name=name + '_out')(x)
  return x


def stack3(x, filters, blocks, stride1=2, groups=32, name=None):
  """A set of stacked residual blocks.

  Arguments:
    x: input tensor.
    filters: integer, filters of the bottleneck layer in a block.
    blocks: integer, blocks in the stacked blocks.
    stride1: default 2, stride of the first layer in the first block.
    groups: default 32, group size for grouped convolution.
    name: string, stack label.

  Returns:
    Output tensor for the stacked blocks.
  """
  x = block3(x, filters, stride=stride1, groups=groups, name=name + '_block1')
  for i in range(2, blocks + 1):
    x = block3( x, filters, groups=groups, conv_shortcut=False, name=name + '_block' + str(i))
  return x

def ResNet50( seq_input ):
  """Instantiates the ResNet50 architecture."""
  def stack_fn(x):
    x = stack1(x, 64, 3, stride1=1, name='conv2')
    x = stack1(x, 128, 4, name='conv3')
    x = stack1(x, 256, 6, name='conv4')
    return stack1(x, 512, 3, name='conv5')
  return ResNet(stack_fn, seq_input, preact=False, use_bias=True, model_name='resnet50')

def ResNet101( seq_input ):
  """Instantiates the ResNet101 architecture."""
  def stack_fn(x):
    x = stack1(x, 64, 3, stride1=1, name='conv2')
    x = stack1(x, 128, 4, name='conv3')
    x = stack1(x, 256, 23, name='conv4')
    return stack1(x, 512, 3, name='conv5')
  return ResNet(stack_fn, seq_input, preact=False, use_bias=True, model_name='resnet101')

def ResNet152( seq_input ):
  """Instantiates the ResNet152 architecture."""
  def stack_fn(x):
    x = stack1(x, 64, 3, stride1=1, name='conv2')
    x = stack1(x, 128, 8, name='conv3')
    x = stack1(x, 256, 36, name='conv4')
    return stack1(x, 512, 3, name='conv5')
  return ResNet(stack_fn, seq_input, preact=False, use_bias=True, model_name='resnet152')

def ResNet50V2( seq_input ):
  """Instantiates the ResNet50V2 architecture."""
  def stack_fn(x):
    x = stack2(x, 64, 3, name='conv2')
    x = stack2(x, 128, 4, name='conv3')
    x = stack2(x, 256, 6, name='conv4')
    return resnet.stack2(x, 512, 3, stride1=1, name='conv5')
  return ResNet( stack_fn, seq_input, preact=True, use_bias=True, model_name='resnet50V2')

def ResNet101V2( seq_input ):
  """Instantiates the ResNet101V2 architecture."""
  def stack_fn(x):
    x = stack2(x, 64, 3, name='conv2')
    x = stack2(x, 128, 4, name='conv3')
    x = stack2(x, 256, 23, name='conv4')
    return stack2(x, 512, 3, stride1=1, name='conv5')
  return ResNet( stack_fn, seq_input, preact=True, use_bias=True, model_name='resnet101V2')

def ResNet152V2( seq_input ):
  """Instantiates the ResNet152V2 architecture."""
  def stack_fn(x):
    x = stack2(x, 64, 3, name='conv2')
    x = stack2(x, 128, 8, name='conv3')
    x = stack2(x, 256, 36, name='conv4')
    return stack2(x, 512, 3, stride1=1, name='conv5')
  return ResNet( stack_fn, seq_input, preact=True, use_bias=True, model_name='resnet152V2')

def ResNeXt50( seq_input ):
  """Instantiates the ResNeXt50 architecture."""
  def stack_fn(x):
    x = stack3(x, filters=128, blocks=3, stride1=1, groups=32, name='conv2')
    x = stack3(x, filters=256, blocks=4, stride1=2, groups=32, name='conv3')
    x = stack3(x, filters=512, blocks=6, stride1=2, groups=32, name='conv4')
    return stack3(x, filters=1024, blocks=3, stride1=2, groups=32, name='conv5')
  return ResNet(stack_fn, seq_input, preact=False, use_bias=False, model_name='resnext50')

def ResNeXt101( seq_input ):
  """Instantiates the ResNeXt101 architecture."""
  def stack_fn(x):
    x = stack3(x, filters=128, blocks=3, stride1=1, groups=32, name='conv2')
    x = stack3(x, filters=256, blocks=4, stride1=2, groups=32, name='conv3')
    x = stack3(x, filters=512, blocks=23, stride1=2, groups=32, name='conv4')
    return stack3(x, filters=1024, blocks=3, stride1=2, groups=32, name='conv5')
  return ResNet(stack_fn, seq_input, preact=False, use_bias=False, model_name='resnext101')
