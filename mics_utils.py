import os
import cv2
import yaml 
import shutil
import numpy as np


def load_config(file_path):
    with open(file_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config


def update_config_file(new_dataset_name, new_class_count, new_input_shape=None, file_path='/Volumes/mydata/projects/lukemia/content/drive/MyDrive/FL_Project/conf/conf.yaml'):
    # Load YAML file
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    # Update values in the YAML data
    data['tabular_mode']['dataset'] = new_dataset_name
    data['tabular_mode']['input_shape'] = new_input_shape
    data['model_params']['class_count'] = new_class_count

    # Save updated YAML file
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)


def calculate_confusion_matrix_metrics(confusion_matrix):
    # Ensure the confusion matrix is square
    if not confusion_matrix.shape[0] == confusion_matrix.shape[1]:
        raise ValueError("Confusion matrix should be a square matrix.")

    num_classes = confusion_matrix.shape[0]

    # Initialize variables to store TP, FP, TN, FN for each class
    tp = np.zeros(num_classes, dtype=int)
    fp = np.zeros(num_classes, dtype=int)
    tn = np.zeros(num_classes, dtype=int)
    fn = np.zeros(num_classes, dtype=int)

    for i in range(num_classes):
        # True positives for class i
        tp[i] = confusion_matrix[i, i]

        # False positives for class i
        fp[i] = np.sum(confusion_matrix[:, i]) - tp[i]

        # False negatives for class i
        fn[i] = np.sum(confusion_matrix[i, :]) - tp[i]

        # True negatives for class i
        tn[i] = np.sum(confusion_matrix) - tp[i] - fp[i] - fn[i]

    return tp, fp, tn, fn


def crop_lukemia_images(src_dir, dst_dir):
    if os.path.exists(dst_dir):
      shutil.rmtree(dst_dir)
    os.mkdir(dst_dir)
    for folder in os.listdir(src_dir):
      os.mkdir(dst_dir+"/"+folder)
      for sub_folder in os.listdir(src_dir+"/"+folder):
          os.mkdir(dst_dir+"/"+folder+"/"+sub_folder)
          for img_file in os.listdir(src_dir+"/"+folder+"/"+sub_folder):
              image = cv2.imread(src_dir+"/"+folder+"/"+sub_folder+"/"+img_file)
              print("Cropping image: ",img_file, image.shape)
              gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
              thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
              result = cv2.bitwise_and(image, image, mask=thresh)
              result[thresh==0] = [255,255,255] 
              (x, y, z_) = np.where(result > 0)
              mnx = (np.min(x))
              mxx = (np.max(x))
              mny = (np.min(y))
              mxy = (np.max(y))
              crop_img = image[mnx:mxx,mny:mxy,:]
              crop_img_r = cv2.resize(crop_img, (224,224))
              cv2.imwrite(dst_dir+"/"+folder+"/"+sub_folder+"/"+img_file, crop_img_r)


def add_noise_to_weights(model, noise_factor=0.01):
    for layer in model.layers:
        if hasattr(layer, 'get_weights'):
            weights = layer.get_weights()
            weights = [w + noise_factor * np.random.normal(size=w.shape) for w in weights]
            layer.set_weights(weights)


def add_gaussian_noise_in_images(images, noise_factor=0.01):
    noisy_images = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    return np.clip(noisy_images, 0.0, 1.0)
    
