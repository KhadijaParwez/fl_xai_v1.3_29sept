import numpy as np
from sklearn.utils import class_weight
from tensorflow import keras
from test_utils import test_scores
from tensorflow.keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping, ModelCheckpoint

def train_our_federated_learninig_model(iterations, clients, edge_servers, global_server, checkpoint_filepath, global_test_gen):
  for i in range(iterations):
    print(f"Starting iteration {i+1}.\n")
    for c in clients:
      connected_edge_server_id = np.random.randint(0,len(edge_servers))
      c.update_model_weights(global_server.global_model.get_weights())
      connected_edge_server = edge_servers[connected_edge_server_id]
      print(f'Training client {c.id}.')
      client_model_weights = c.train_local_model()
      print(f"Weight updates from local model of clinet {c.id} are updated to edge server {connected_edge_server_id+1}.")
      connected_edge_server.client_model_weights.append(client_model_weights)
      print()

    print(f"Performing weight aggregations on edge servers.")
    for es in edge_servers:
      if len(es.client_model_weights) == 1:
        es_weights = es.client_model_weights
        global_server.edge_server_weights.append(es_weights)
        es.client_model_weights = []
        es.aggregated_model_weights = []

      if len(es.client_model_weights) > 1:
        es_weights = es.aggregate_model_weights()
        global_server.edge_server_weights.append(es_weights)
        es.client_model_weights = []
        es.aggregated_model_weights = []

    print("")
    print(f"Performing weight aggregations on global server.")
    global_server.aggregate_model_weights()
    print(f"Updating weights of global model.")
    global_server.update_global_model()
    print("Evaluating global model:\n")

    test_loss, test_accuracy, test_precision, test_recall, test_f1 = global_server.global_model.evaluate(global_test_gen)
    print(f"Iteration {i + 1}: Test Accuracy: {test_accuracy * 100:.2f}%, Test Precision: {test_precision * 100:.2f}%, Test Recall: {test_recall * 100:.2f}%, Test F1 Score: {test_f1 * 100:.2f}%")

    global_server.global_model.save(f"{checkpoint_filepath}/fl_global_model")
    global_server.aggregated_model_weights = []
    global_server.edge_server_weights = []


    print(f"\nIteration {i+1} completed successfully.")
    print('\n\n')

def train_traditional_federated_learninig_model(iterations, datasets, client_models, global_model, checkpoint_filepath, epochs, global_test_gen, logs_filepath, class_weight_dict):
  
  for i in range(iterations):
    print(f"Iteration {i + 1}/{iterations}")

    client_weights = []
    for c in range(len(datasets)):
      checkpoint_filepath = f"{checkpoint_filepath}/{c}"
      tensorboard_callback = TensorBoard(log_dir=f"{logs_filepath}/{c}")
      model_checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
      callbacks = [model_checkpoint, tensorboard_callback]
      print(f"client {c+1}")
      train_gen, valid_gen, test_gen = datasets[c]
      model = client_models[c]
      history = model.fit(x=train_gen, epochs=epochs, verbose=1, validation_data=valid_gen, callbacks=callbacks, class_weight=class_weight_dict)
      client_weights.append(model.get_weights())

    global_weights = np.mean(client_weights, axis=0)

    # Update global model with aggregated weights
    global_model.set_weights(global_weights)
    global_model.save_weights(f"{checkpoint_filepath}/fl_global_model")

    # Evaluate global model on test data
    test_loss, test_accuracy, test_precision, test_recall, test_f1 = global_model.evaluate(global_test_gen)
    print(f"Iteration {i + 1}: Test Accuracy: {test_accuracy * 100:.2f}%, Test Precision: {test_precision * 100:.2f}%, Test Recall: {test_recall * 100:.2f}%, Test F1 Score: {test_f1 * 100:.2f}%")

    # Update client models with global model weights
    for c in range(len(datasets)):
        client_models[c].set_weights(global_weights)


def train_individual_clients(datasets, client_models, epochs, checkpoint_filepath, patience, logs_filepath, class_weight_dict):
  
  for i in range(len(datasets)):
    checkpoint_filepath = f"{checkpoint_filepath}/{i}"
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, restore_best_weights=True)
    tensorboard_callback = TensorBoard(log_dir=logs_filepath)
    model_checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    callbacks = [early_stopping, model_checkpoint, tensorboard_callback]
    data = datasets[i]
    model = client_models[i]
    train_gen, valid_gen, test_gen = data
    
    history = client_models[i].fit(x=train_gen, epochs=epochs, verbose=1, validation_data=valid_gen, callbacks=callbacks, class_weight=class_weight_dict)
    
    ts_length = len(test_gen.classes)
    test_batch_size = test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length%n == 0 and ts_length/n <= 80]))
    test_steps = ts_length // test_batch_size

    train_score = model.evaluate(train_gen, steps= test_steps, verbose= 1)
    valid_score = model.evaluate(valid_gen, steps= test_steps, verbose= 1)
    test_score = model.evaluate(test_gen, steps= test_steps, verbose= 1)

    print("Train Loss: ", train_score[0])
    print("Train Accuracy: ", train_score[1])
    print('-' * 20)
    print("Validation Loss: ", valid_score[0])
    print("Validation Accuracy: ", valid_score[1])
    print('-' * 20)
    print("Test Loss: ", test_score[0])
    print("Test Accuracy: ", test_score[1])

    test_scores(test_gen, model)
    