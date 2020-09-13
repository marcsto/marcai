import tensorflow as tf
from kerastuner import HyperModel


class FullyConnectedHpModel(HyperModel):
  def __init__(self, classes, loss, input_layer):
    self.classes = classes
    self.loss = loss
    self.input_layer = input_layer
    # tf.keras.losses.BinaryCrossentropy(from_logits=True),
    # 'sparse_categorical_crossentropy'

  def build(self, hp):
    model = tf.keras.Sequential()
    #inputs = tf.keras.utils.get_source_inputs(self.input_tensor)
    #x = self.input_tensor
    model.add(self.input_layer)
            
    #inputs = layers.Input(shape=self.input_shape)
    for _ in range(hp.Int('layers', 1, 5, default=1)):
      model.add(tf.keras.layers.Dense(units=hp.Int('units',
                                            min_value=32,
                                            max_value=512,
                                            step=32),
                               activation='relu'))
    model.add(tf.keras.layers.Dense(self.classes))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice('learning_rate',
                      values=[1e-2, 1e-3, 1e-4])),
        loss=self.loss,
        metrics=['accuracy'])
    return model
      
def fully_connected(inputs, output_count, layer_count=1, units_per_layer=256):
  model = tf.keras.Sequential()
  model.add(inputs)
  #model.add(tf.keras.layers.Flatten())
  for _ in range(layer_count):
    model.add(tf.keras.layers.Dense(units_per_layer, activation='relu'))
  model.add(tf.keras.layers.Dense(output_count))
  
  return model
  
def compile_binary(model, optimizer='adam', metrics=['accuracy']):
  model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=optimizer,
    metrics=metrics)