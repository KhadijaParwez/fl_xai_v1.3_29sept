# import Deep learning Libraries
import tensorflow as tf
from mics_utils import load_config
from tensorflow import keras
from keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization

def keras_recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def keras_precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def keras_f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1

def get_base_model(model_name, img_shape):
    try:
        model_class = getattr(keras.applications, model_name)
    except AttributeError:
        raise ValueError(f"Invalid model name '{model_name}'. Model not found.")
    
    base_model = model_class(
        include_top=False, weights="imagenet", input_shape=img_shape, pooling='max'
    )    
    return base_model

def get_model(config_file='/Volumes/mydata/projects/lukemia/content/drive/MyDrive/FL_Project/conf/conf.yaml'):

    # Load configuration from YAML file
    config = load_config(config_file)
    tabular = config['tabular_mode']['dataset'] == 'HIV'
    # Extract parameters
    img_shape = (
        config['model_params']['img_shape']['height'],
        config['model_params']['img_shape']['width'],
        config['model_params']['img_shape']['channels']
    )
    backbone_model = config['model_params']['backbone_model_name']
    freeze_base_model = config['model_params']['freeze_base_model']
    class_count = config['model_params']['class_count']
    learning_rate = config['model_params']['learning_rate']
    momentum = config['model_params']['momentum']
    epsilon = config['model_params']['epsilon']
    dense_units = config['model_params']['dense_units']
    dropout_rate = config['model_params']['dropout_rate']
    l2_regularizer = config['model_params']['l2_regularizer']
    l1_regularizer = config['model_params']['l1_regularizer']
    input_shape = config['tabular_mode']['input_shape']

    if tabular:
        # Build the Keras model
        model = Sequential()
        model.add(Dense(64, input_dim=input_shape, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        optimizer = getattr(keras.optimizers, config['training_params']['optimizer'])
        loss = config['training_params']['loss']
        metrics = [
            'accuracy',
            keras_precision,
            keras_recall,
            keras_f1_score
        ]
        model.compile(optimizer(learning_rate=learning_rate), loss=loss, metrics=metrics)
        return model
    else:
        # create pre-trained model (you can build on a pre-trained model such as EfficientNet, VGG, ResNet)
        base_model = get_base_model(backbone_model, img_shape)
        if freeze_base_model:
            base_model.trainable = False

        model = Sequential([
            base_model,
            BatchNormalization(axis=-1, momentum=momentum, epsilon=epsilon),
            Dense(
                dense_units,
                kernel_regularizer=regularizers.l2(l=l2_regularizer),
                activity_regularizer=regularizers.l1(l=l1_regularizer),
                bias_regularizer=regularizers.l1(l=l1_regularizer),
                activation='relu'
            ),
            Dropout(rate=dropout_rate, seed=123),
            Dense(class_count, activation='softmax')
        ])

        optimizer = getattr(keras.optimizers, config['training_params']['optimizer'])
        loss = config['training_params']['loss']
        metrics = [
            'accuracy',
            keras_precision,
            keras_recall,
            keras_f1_score
        ]

        model.compile(optimizer(learning_rate=learning_rate), loss=loss, metrics=metrics)

        return model
