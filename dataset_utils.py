import os 
import shutil
import numpy as np
import pandas as pd 
from zipfile import ZipFile
import matplotlib.pyplot as plt
from mics_utils import load_config
from sklearn.impute import SimpleImputer
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Generate data paths with labels
def define_paths(dir, dataset_type):
    filepaths = []
    labels = []
    folds = os.listdir(dir)

    for fold in folds:
        foldpath = os.path.join(dir, fold)
        filelist = os.listdir(foldpath)
        if dataset_type == 1:
            for fold_ in filelist:
                foldpath_ = os.path.join(foldpath, fold_)
                filelist_ = os.listdir(foldpath_)

                for file_ in filelist_:
                    fpath = os.path.join(foldpath_, file_)
                    filepaths.append(fpath)
                    labels.append(fold_)

        elif dataset_type == 2:
          for file in filelist:
              fpath = os.path.join(foldpath, file)
              filepaths.append(fpath)
              labels.append(fold)

    return filepaths, labels


# Concatenate data paths with labels into one dataframe ( to later be fitted into the model )
def define_df(files, classes):
    Fseries = pd.Series(files, name= 'filepaths')
    Lseries = pd.Series(classes, name='labels')
    return pd.concat([Fseries, Lseries], axis= 1)


# Function that create dataframe for train, validation, and test data
def create_df(data_dir, dataset_type):

    # train dataframe
    files, classes = define_paths(data_dir, dataset_type)
    df = define_df(files, classes)

    strat = df['labels']
    train_df, dummy_df = train_test_split(df, train_size=0.9, shuffle=True, random_state=123, stratify=strat)

    # test dataframe
    strat = dummy_df['labels']
    valid_df, test_df= train_test_split(dummy_df, train_size=0.3, shuffle=True, random_state=123, stratify=strat)

    return train_df, valid_df, test_df


def create_model_data (train_df, valid_df, test_df, batch_size, config_file='/Volumes/mydata/projects/lukemia/content/drive/MyDrive/FL_Project/conf/conf.yaml'):
    '''
    This function takes train, validation, and test dataframe and fit them into image data generator, because model takes data from image data generator.
    Image data generator converts images into tensors. '''
    config = load_config(config_file)
    img_height = config['model_params']['img_shape']['height']
    img_width = config['model_params']['img_shape']['width']
    img_channels = config['model_params']['img_shape']['channels']
    featurewise_center = config['augmentation_params']['featurewise_center']
    samplewise_center = config['augmentation_params']['samplewise_center']
    featurewise_std_normalization = config['augmentation_params']['featurewise_std_normalization']
    samplewise_std_normalization = config['augmentation_params']['samplewise_std_normalization']
    zca_whitening = config['augmentation_params']['zca_whitening']
    zca_epsilon = config['augmentation_params']['zca_epsilon']
    rotation_range = config['augmentation_params']['rotation_range']
    width_shift_range = config['augmentation_params']['width_shift_range']
    height_shift_range = config['augmentation_params']['height_shift_range']
    brightness_range = config['augmentation_params']['brightness_range']
    shear_range = config['augmentation_params']['shear_range']
    zoom_range = config['augmentation_params']['zoom_range']
    channel_shift_range = config['augmentation_params']['channel_shift_range']
    fill_mode = config['augmentation_params']['fill_mode']
    cval = config['augmentation_params']['cval']
    horizontal_flip = config['augmentation_params']['horizontal_flip']
    vertical_flip = config['augmentation_params']['vertical_flip']
    rescale = config['augmentation_params']['rescale']
    interpolation_order = config['augmentation_params']['interpolation_order']
    augment_training_data = config['data_params']['augment_training_data']
    augment_test_data = config['data_params']['augment_test_data']

    img_shape = (
      img_height, 
      img_width,
      img_channels
    )
    img_size = (
      img_height, 
      img_width,
    )
    if img_channels == 3:
      color = 'rgb'
    else:
      color = 'gray'

    # Recommended : use custom function for test data batch size, else we can use normal batch size.
    ts_length = len(test_df)
    test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length%n == 0 and ts_length/n <= 80]))
    test_steps = ts_length // test_batch_size

    # This function which will be used in image data generator for data augmentation, it just take the image and return it again.
    def scalar(img):
        return img

    if augment_training_data:
      tr_gen = ImageDataGenerator(
        preprocessing_function=scalar, 
        featurewise_center=featurewise_center,
        samplewise_center=samplewise_center,
        featurewise_std_normalization=featurewise_std_normalization,
        samplewise_std_normalization=samplewise_std_normalization,
        zca_whitening=zca_whitening,
        zca_epsilon=zca_epsilon,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        brightness_range=brightness_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        channel_shift_range=channel_shift_range,
        fill_mode=fill_mode,
        cval=cval,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        rescale=rescale,
        interpolation_order=interpolation_order
      )
    else:
      tr_gen = ImageDataGenerator(preprocessing_function=scalar, horizontal_flip=True)
    if augment_test_data:
      ts_gen = ImageDataGenerator(
          preprocessing_function=scalar, 
          featurewise_center=featurewise_center,
          samplewise_center=samplewise_center,
          featurewise_std_normalization=featurewise_std_normalization,
          samplewise_std_normalization=samplewise_std_normalization,
          zca_whitening=zca_whitening,
          zca_epsilon=zca_epsilon,
          rotation_range=rotation_range,
          width_shift_range=width_shift_range,
          height_shift_range=height_shift_range,
          brightness_range=brightness_range,
          shear_range=shear_range,
          zoom_range=zoom_range,
          channel_shift_range=channel_shift_range,
          fill_mode=fill_mode,
          cval=cval,
          horizontal_flip=horizontal_flip,
          vertical_flip=vertical_flip,
          rescale=rescale,
          interpolation_order=interpolation_order
        )
    else:
      ts_gen = ImageDataGenerator(preprocessing_function=scalar)

    train_gen = tr_gen.flow_from_dataframe( train_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= color, shuffle= True, batch_size= batch_size)

    valid_gen = ts_gen.flow_from_dataframe( valid_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= color, shuffle= True, batch_size= batch_size)

    # Note: we will use custom test_batch_size, and make shuffle= false
    test_gen = ts_gen.flow_from_dataframe( test_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= color, shuffle= False, batch_size= test_batch_size)

    return train_gen, valid_gen, test_gen


def show_images(gen):
    '''
    This function take the data generator and show sample of the images
    '''

    # return classes , images to be displayed
    g_dict = gen.class_indices        # defines dictionary {'class': index}
    classes = list(g_dict.keys())     # defines list of dictionary's kays (classes), classes names : string
    images, labels = next(gen)        # get a batch size samples from the generator

    # calculate number of displayed samples
    length = len(labels)        # length of batch size
    sample = min(length, 25)    # check if sample less than 25 images

    plt.figure(figsize= (20, 20))

    for i in range(sample):
        plt.subplot(5, 5, i + 1)
        image = images[i] / 255       # scales data to range (0 - 255)
        plt.imshow(image)
        index = np.argmax(labels[i])  # get image index
        class_name = classes[index]   # get class of image
        plt.title(class_name, color= 'blue', fontsize= 12)
        plt.axis('off')
    plt.show()


def preprocess_hiv_data(hiv_data):
    # Assuming your DataFrame is named 'df'
    # Drop irrelevant columns
    columns_to_drop = ['PLHIVID', 'City', 'LatestPPTCTStatus', 'PEPStatus']
    df = hiv_data.drop(columns=columns_to_drop)

    # Convert date columns to datetime format
    date_columns = ['ARTInitiatedDate', 'LastViralLoadDate', 'LastARTIssuanceDate']
    df[date_columns] = df[date_columns].apply(pd.to_datetime)

    df['TreatmentDuration'] = (df['LastARTIssuanceDate'] - df['ARTInitiatedDate']).dt.days


    # Handle missing values
    # Impute missing values in numerical columns using median
    numerical_columns = df.select_dtypes(include=['number']).columns
    df[numerical_columns] = SimpleImputer(strategy='median').fit_transform(df[numerical_columns])

    # Impute missing values in categorical columns using mode
    categorical_columns = df.select_dtypes(include=['object']).columns
    df[categorical_columns] = SimpleImputer(strategy='most_frequent').fit_transform(df[categorical_columns])

    # Convert categorical variables
    df = pd.get_dummies(df, columns=['Gender', 'RegistrationStatus', 'Typology', 'ModeOfTransmission', 'DeathStatus'], drop_first=True)

    # Handle date columns

    # Age transformation
    bins = [0, 18, 35, 50, 100]
    labels = ['0-18', '19-35', '36-50', '51-100']
    df['AgeGroup'] = pd.cut(df['AgeinYears'], bins=bins, labels=labels, include_lowest=True)


    # One-hot encoding for 'DistrictName'
    top_districts = df['DistrictName'].value_counts().nlargest(10).index
    df['DistrictTop'] = df['DistrictName'].apply(lambda x: x if x in top_districts else 'Other')
    df = pd.get_dummies(df, columns=['DistrictTop'], prefix='District', drop_first=True)
    df = df.drop(columns=['DistrictName'])

    # Selected set of drugs to create binary columns
    selected_drugs = ['Dolutegravir', 'Lamivudine', 'Tenofovir', 'Efavirenz', 'Zidovudine', 'Abacavir', 'Nevirapine', 'Lopinavir', 'Ritonavir']

    # Create binary columns for selected drugs
    for drug in selected_drugs:
        df[f'{drug}_Present'] = df['LastRegimen'].str.contains(drug).astype(int)


    # One-hot encoding for 'AgeGroup'
    df = pd.get_dummies(df, columns=['AgeGroup'], prefix='Age', drop_first=True)
    class_labels = df['LTFPStatus'].unique()
    df = df.drop(columns=date_columns+['AgeinYears']+['LastRegimen'])
    df = df.rename(columns={'LTFPStatus': 'labels'})

    return df, class_labels


class DataFrameGenerator(Sequence):
    def __init__(self, df_features, df_labels, batch_size, scaler=None, label_binarizer=None):
        self.df_features = df_features
        self.df_labels = df_labels
        self.batch_size = batch_size
        self.scaler = scaler
        self.label_binarizer = label_binarizer
        self.indexes = np.arange(len(self.df_features))
        
        # Initialize classes and class_indices
        self.classes = None
        self.class_indices = None

        # Populate classes and class_indices
        self._initialize_classes()

    def _initialize_classes(self):
        # Unique binary labels
        unique_classes = np.unique(self.df_labels)
        self.class_indices = {label: i for i, label in enumerate(unique_classes)}
        self.classes = [self.class_indices[label] for label in self.df_labels]

    def __len__(self):
        return int(np.ceil(len(self.df_features) / self.batch_size))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_features = self.df_features.iloc[batch_indexes]
        batch_labels = self.df_labels.iloc[batch_indexes]

        # Convert features and labels to NumPy arrays
        X_batch = batch_features.values
        y_batch = batch_labels.values

        # Apply standard scaling
        if self.scaler is not None:
            X_batch = self.scaler.transform(X_batch)

        # Binarize labels
        if self.label_binarizer is not None:
            y_batch = self.label_binarizer.transform(y_batch)

        return X_batch, y_batch

def create_data_generators(train_df, valid_df, test_df, batch_size):
    # Separate features and labels
    train_features = train_df.drop(columns=['labels'])
    train_labels = train_df['labels']

    valid_features = valid_df.drop(columns=['labels'])
    valid_labels = valid_df['labels']

    test_features = test_df.drop(columns=['labels'])
    test_labels = test_df['labels']

    # Initialize and fit a scaler on the training data
    scaler = StandardScaler()
    scaler.fit(train_features)

    # Initialize and fit a label binarizer on the training labels
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(train_labels)

    # Create generators for training, validation, and testing
    train_generator = DataFrameGenerator(train_features, train_labels, batch_size, scaler, label_binarizer)
    valid_generator = DataFrameGenerator(valid_features, valid_labels, batch_size, scaler, label_binarizer)
    test_generator = DataFrameGenerator(test_features, test_labels, batch_size, scaler, label_binarizer)

    return train_generator, valid_generator, test_generator
