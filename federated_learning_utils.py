import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
import warnings
import os
warnings.filterwarnings('ignore')
from modeling_utils import get_model

class Client:

  def __init__(self, id, dataset, model_save_path, logs_filepath, train_epochs, class_weight_dict):

    self.id = id
    self.connected_client = None
    self.train_gen = dataset[0]
    self.valid_gen = dataset[1]
    self.test_gen = dataset[2]
    self.model_save_path = f"{model_save_path}/{self.id}"
    self.train_epochs = train_epochs
    self.logs_filepath = logs_filepath
    self.class_weight_dict = class_weight_dict
    self.callbacks = [keras.callbacks.ModelCheckpoint(self.model_save_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1), 
                      TensorBoard(log_dir=f"{self.logs_filepath}/{self.id}")]
    self.model = self.initialize_model()

  def train_local_model(self):
    self.model.fit(x = self.train_gen, validation_data = self.valid_gen, epochs=self.train_epochs, callbacks=self.callbacks, verbose=1, class_weight=self.class_weight_dict)
    return self.model.get_weights()

  def update_model_weights(self, weights):
    self.model.set_weights(weights)

  def connect_other_clients(self,client):
    client.connected_client = self

  def initialize_model(self):

    model = get_model()

    if os.path.exists(self.model_save_path):
      print('Loading saved checkpoint')
      model.load_weights(self.model_save_path)
    print(f"Model initialized successfully for client {self.id}.")
    print()

    return model

class EdgeServer:

  def __init__(self, id):
    self.id = id
    self.client_model_weights = []
    self.aggregated_model_weights = []

  def aggregate_model_weights(self, mode = 'avg'):
    if mode == 'avg':
      self.aggregated_model_weights = np.mean(self.client_model_weights, axis=0)

    else:
      print("Selcted aggregation mode is not supported.")

    return self.aggregated_model_weights


class GlobalServer:
  def __init__(self):
    self.edge_server_weights = []
    self.aggregated_model_weights = []
    self.global_model = self.initialize_global_model()

  def aggregate_model_weights(self, mode = 'avg'):
    if mode == 'avg':
      self.aggregated_model_weights = np.mean(self.edge_server_weights, axis=0)

    else:
      print("Selcted aggregation mode is not supported.")

    return self.aggregated_model_weights

  def initialize_global_model(self):

    model = get_model()

    print(f"Global model initialized successfully.")
    print()

    return model

  def update_global_model(self):

    if len(self.aggregated_model_weights) == len(self.global_model.get_weights()):
      self.global_model.set_weights(self.aggregated_model_weights)
    elif len(self.aggregated_model_weights) == 1:
      self.global_model.set_weights(self.aggregated_model_weights[0])
    else:
      raise Exception(str('Model weights not updated!'+ str(len(self.aggregated_model_weights))))
      