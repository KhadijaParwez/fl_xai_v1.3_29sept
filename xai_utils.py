import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries
from jax import numpy as jnp
import shap
import numpy as np
import matplotlib.cm as cm
from tensorflow import keras
from skimage.segmentation import slic
from lime import lime_image
from mics_utils import calculate_confusion_matrix_metrics
from tf_keras_vis.gradcam import Gradcam
from tensorflow.keras import backend as K
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from matplotlib.gridspec import GridSpec
from tf_keras_vis.utils.scores import CategoricalScore


# Function to create a modular Saliency model
def create_saliency_model(model, img, replace2linear, score):
        
    saliency = Saliency(model, model_modifier=replace2linear, clone=True)
    smooth_grad = saliency(score, img, smooth_samples=20, smooth_noise=0.20)
    saliency_map = saliency(score, img)
    
    return saliency_map[0], smooth_grad[0]

# Function to create a modular Gradcam model
def create_gradcam_model(model, replace2linear, score, img, efficientnet_model, layer_id):
    gradcam = Gradcam(efficientnet_model, model_modifier=replace2linear, clone=True)
    gradcam_cam = gradcam(score, img, penultimate_layer=layer_id)
    grad_cam_heatmap = np.uint8(cm.jet(gradcam_cam[0])[..., :3] * 255)
    return grad_cam_heatmap

# Function to create a modular Gradcam++ model
def create_gradcam_plus_plus_model(model, replace2linear, score, img, efficientnet_model, layer_id):
    gradcam_plus_plus = GradcamPlusPlus(efficientnet_model, model_modifier=replace2linear, clone=True)
    gradcam_plus_plus_cam = gradcam_plus_plus(score, img, penultimate_layer=layer_id)
    gradcam_plus_plus_heatmap = np.uint8(cm.jet(gradcam_plus_plus_cam[0])[..., :3] * 255)
    return gradcam_plus_plus_heatmap

# Function to create a modular Scorecam model
def create_scorecam_model(model, efficientnet_model, score, img, layer_id):
    scorecam = Scorecam(efficientnet_model)
    scorecam_cam = scorecam(score, img, penultimate_layer=layer_id)
    scorecam_cam_heatmap = np.uint8(cm.jet(scorecam_cam[0])[..., :3] * 255)
    return scorecam_cam_heatmap

# Function to create a modular Faster Scorecam model
def create_faster_scorecam_model(model, replace2linear, score, img, efficientnet_model, layer_id):
    faster_scorecam = Scorecam(efficientnet_model, model_modifier=replace2linear)
    faster_scorecam_cam = faster_scorecam(score, img, penultimate_layer=layer_id, max_N=10)
    faster_scorecam_heatmap = np.uint8(cm.jet(faster_scorecam_cam[0])[..., :3] * 255)
    return faster_scorecam_heatmap

# Function to create modular Lime explanation
def create_lime_explanation(img, class_labels, model, explainer):
    lime_img = lime_explanation(img, class_labels, model, explainer)[0]
    return lime_img

# Function to create modular Shap explanation
def create_shap_explanation(img, model, explainer):
    shap_explaination(img, model, explainer)

def create_heatmap_and_superimpose(img, efficientnet_model, layer_name):
    hires_heatmap = generate_cam(img, efficientnet_model, layer_name)
    respond_heatmap = generate_cam(img, efficientnet_model, layer_name, respond=True)

    resized_hires_heatmap = resize_heatmap(img, hires_heatmap)
    resized_respond_heatmap = resize_heatmap(img, respond_heatmap)

    superimposed_hires_img = superimpose_heatmap(img[0], resized_hires_heatmap)
    superimposed_respond_img = superimpose_heatmap(img[0], resized_respond_heatmap)

    return superimposed_hires_img, superimposed_respond_img, resized_hires_heatmap, resized_respond_heatmap


def combine_heatmaps(heatmaps_dict):

    # Sorting methods based on intensity levels
    sorted_methods = sorted(heatmaps_dict.items(), key=lambda x: x[1][0])

    # Normalize intensity levels to [0, 1]
    max_intensity = max([method[1][0] for method in sorted_methods])
    normalized_intensities = [method[1][0] / max_intensity for method in sorted_methods]

    # Weighted combination of heatmaps based on intensity levels
    combined_heatmap = np.zeros_like(sorted_methods[0][1][1], dtype=float)
    for i, (_, (intensity, heatmap)) in enumerate(sorted_methods):
        combined_heatmap += normalized_intensities[i] * heatmap

    # Normalize combined heatmap to [0, 1]
    combined_heatmap /= np.max(combined_heatmap)

    return combined_heatmap


def predict_and_display(image, model, class_labels, plot_xai=False, xai_heatmap=None):
    # Make a prediction using the model
    probabilities = model.predict(np.expand_dims(image, axis=0))[0]

    # Get the predicted class index with the highest probability
    predicted_class_idx = np.argmax(probabilities)
    predicted_class_label = class_labels[predicted_class_idx]
    confidence = probabilities[predicted_class_idx] * 100

    # Display the image
    plt.subplot(1, 2, 1)
    plt.imshow(image / 255)
    plt.axis('off')

    # Add the prediction text inside the image
    text = f"This sample most likely belongs to '{predicted_class_label}' \n with {confidence:.2f}% confidence."
    plt.text(0, image.shape[0] + 30, text, fontsize=9, color='black', backgroundcolor='white')

    # Display other class predictions
    other_classes_text = ""
    for i, prob in enumerate(probabilities):
        if i != predicted_class_idx:
            class_label = class_labels[i]
            confidence = prob * 100
            other_classes_text += f"\n- {class_label}: {confidence:.2f}%"

    # Add the other class predictions text inside the image
    plt.text(0, image.shape[0] + 60, "Other classes:" + other_classes_text, fontsize=12, color='black',
             backgroundcolor='white')

    # Plot XAI heatmap if requested
    if plot_xai and xai_heatmap is not None:
        plt.subplot(1, 2, 2)
        plt.imshow(xai_heatmap) 
        plt.axis('off')
        plt.title("Combined Heatmap")

    plt.show()

def display_images_in_grid(images, title):

    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    fig.suptitle(title, fontsize=16)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i]/255)
        ax.axis("off")
        ax.set_title(f"Image {i+1}")

    plt.tight_layout()
    plt.show()


# Define a function to get the model gradients
def gradient(images, labels, model):
  images = tf.Variable(images)
  with tf.GradientTape() as tape:
    tape.watch(images)
    preds = model(images)
    loss = tf.reduce_sum(preds * labels)
  grads = tape.gradient(loss, images)
  return grads.numpy()

# Define a function to get the LIME explanations
def explain(images, predict, num_samples=1000, num_features=10):
  explainer = lime_image.LimeImageExplainer()
  explanations = []
  for image in images:
    segments = slic(image, n_segments=num_features, compactness=1, sigma=1)
    explanation = explainer.explain_instance(image, predict, top_labels=1, hide_color=0, num_samples=num_samples, segmentation_fn=lambda x: segments)
    explanations.append(explanation)
  return explanations

# Define a function to get the LIME gradients
def lime_gradient(images, explanations):
  grads = []
  for image, explanation in zip(images, explanations):
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False)
    mask = np.expand_dims(mask, axis=-1)
    grad = (image - temp) * mask
    grads.append(grad)
  return np.array(grads)

# Define a function to calculate the infidelity score
def infidelity_score(model_grads, lime_grads):
  score = 0
  if len(model_grads) > 0:
    for mg, lg in zip(model_grads, lime_grads):
      score += np.abs(np.sum(mg * lg)) / (np.linalg.norm(mg) * np.linalg.norm(lg) + 0.0001)
    score /= len(model_grads)
    return score
  else:
    print('No grad detected')

# Define a function to calculate the sensitivity score
def sensitivity_score(explanations, predict, num_perturbations=1, epsilon=0.01):
  score = 0
  for explanation in explanations:
    image, _ = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False)
    perturbations = np.random.normal(0, epsilon, size=(num_perturbations,) + image.shape)
    perturbed_images = image + perturbations
    perturbed_explanations = explain(perturbed_images, predict)
    perturbed_weights = np.array([explanation.local_exp[explanation.top_labels[0]] for explanation in perturbed_explanations])
    original_weights = np.array(explanation.local_exp[explanation.top_labels[0]])
    score += np.var(perturbed_weights - original_weights) / np.var(original_weights)
  score /= len(explanations)
  return score


# Calculate Infidelity
def calculate_infidelity(original_model, surrogate_model, test_data):
    original_predictions = original_model.predict(test_data)
    surrogate_predictions = surrogate_model.predict(test_data)

    # Convert the predictions to TensorFlow tensors
    original_predictions_tensor = tf.convert_to_tensor(original_predictions, dtype=tf.float32)
    surrogate_predictions_tensor = tf.convert_to_tensor(surrogate_predictions, dtype=tf.float32)

    # Choose an appropriate evaluation metric, e.g., mean squared error
    infidelity_metric = keras.metrics.mean_squared_error(original_predictions_tensor, surrogate_predictions_tensor)

    infidelity = infidelity_metric.numpy()
    return infidelity



# Calculate Sensitivity
def calculate_sensitivity(interpretation_model, sample, perturbation_scale=0.1):
    perturbed_samples = []

    # Apply perturbation to each feature
    for i in range(len(sample)):
        perturbed_sample = np.copy(sample)
        perturbed_sample[i] += perturbation_scale * np.random.randn(*perturbed_sample[i].shape)
        perturbed_samples.append(perturbed_sample)

    perturbed_samples = np.array(perturbed_samples)

    # Interpret original sample and perturbed samples
    original_interpretation = interpretation_model.predict(sample[np.newaxis])
    perturbed_interpretations = interpretation_model.predict(perturbed_samples)

    # Calculate sensitivity score based on how much interpretation changes
    sensitivity = np.mean(np.square(original_interpretation - perturbed_interpretations))
    return sensitivity


def get_images_for_xai_binary(datasets, model, confusion_matrix, tp_limit = 4, tn_limit = 4, fp_limit = 4, fn_limit = 4):

    # Initialize lists to store images for each category
    tp_images = []
    tn_images = []
    fp_images = []
    fn_images = []

    tp, fp, tn, fn = calculate_confusion_matrix_metrics(confusion_matrix)    
    tp_limit = min(max(tp),tp_limit)
    fp_limit = min(max(fp),fp_limit)
    tn_limit = min(max(tn),tn_limit)
    fn_limit = min(max(fn),fn_limit)
    
    # Loop through the test data generator
    for batch in datasets[0][2]:
        images = batch[0]  # Extract the images from the batch
        labels = batch[1]  # Extract the ground truth labels from the batch

        # Make predictions using the model
        predictions = model.predict(images, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)  # Assuming you are using one-hot encoded labels

        # Iterate through each image and its predicted and true label
        for i in range(len(images)):
            image = images[i]
            true_label = np.argmax(labels[i])  # Convert one-hot encoded label back to an integer

            # Get the predicted label for the image
            predicted_label = predicted_labels[i]

            # Determine TP, TN, FP, FN and store images accordingly
            if true_label == 0 and predicted_label == 0 and len(tp_images) < tp_limit:  # True Positive
                tp_images.append(image)
            elif true_label == 1 and predicted_label == 1 and len(tn_images) < tn_limit:  # True Negative
                tn_images.append(image)
            elif true_label == 1 and predicted_label == 0 and len(fp_images) < fp_limit:  # False Positive
                fp_images.append(image)
            elif true_label == 0 and predicted_label == 1 and len(fn_images) < fn_limit:  # False Negative
                fn_images.append(image)

            # If you have collected the specified number of images for each category, break the loop
            if len(tp_images) == tp_limit and len(tn_images) == tn_limit and len(fp_images) == fp_limit and len(fn_images) == fn_limit:
                break

        # If you have collected the specified number of images for each category, break the loop
        if len(tp_images) == tp_limit and len(tn_images) == tn_limit and len(fp_images) == fp_limit and len(fn_images) == fn_limit:
            break

    true_pos_images = np.array(tp_images)
    true_neg_images = np.array(tn_images)
    false_pos_images = np.array(fp_images)
    false_neg_images = np.array(fn_images)

    return true_pos_images, true_neg_images, false_pos_images, false_neg_images


def get_images_for_xai_multi(datasets, model, class_count, confusion_matrix, tp_limit = 4, tn_limit = 4, fp_limit = 4, fn_limit = 4):

  # Initialize lists to store images for each category
  tp_images = []
  tn_images = []
  fp_images = []
  fn_images = []
  
  tp, fp, tn, fn = calculate_confusion_matrix_metrics(confusion_matrix)
  tp_limit = min(max(tp),tp_limit)
  fp_limit = min(max(fp),fp_limit)
  tn_limit = min(max(tn),tn_limit)
  fn_limit = min(max(fn),fn_limit)

  # Loop through the test data generator
  for batch in datasets[0][2]:
      images = batch[0]  # Extract the images from the batch
      labels = batch[1]  # Extract the ground truth labels from the batch

      # Make predictions using the model
      predictions = model.predict(images, verbose=0)
      predicted_labels = np.argmax(predictions, axis=1)  # Assuming you are using one-hot encoded labels

      # Iterate through each image and its predicted and true label
      for i in range(len(images)):
          image = images[i]
          true_label = np.argmax(labels[i])  # Convert one-hot encoded label back to an integer

          # Get the predicted label for the image
          predicted_label = predicted_labels[i]

          # Determine TP, TN, FP, FN and store images accordingly
          for class_index in range(class_count):
              if true_label == class_index and predicted_label == class_index and len(tp_images) < tp_limit:  # True Positive
                  tp_images.append(image)
              elif true_label == class_index and predicted_label != class_index and len(fn_images) < fn_limit:  # False Negative
                  fn_images.append(image)
              elif true_label != class_index and predicted_label == class_index and len(fp_images) < fp_limit:  # False Positive
                  fp_images.append(image)
              elif true_label != class_index and predicted_label != class_index and len(tn_images) < tn_limit:  # True Negative
                  tn_images.append(image)

          # If you have collected 4 images for each category, break the loop
          if len(tp_images) == tp_limit and len(tn_images) == tn_limit and len(fp_images) == fp_limit and len(fn_images) == fn_limit:
              break
          # print(len(tp_images), len(tn_images), len(fp_images),len(fn_images))
      # If you have collected 4 images for each category, break the loop
      if len(tp_images) == tp_limit and len(tn_images) == tn_limit and len(fp_images) == fp_limit and len(fn_images) == fn_limit:
          break

  true_pos_images = np.array(tp_images)
  true_neg_images = np.array(tn_images)
  false_pos_images = np.array(fp_images)
  false_neg_images = np.array(fn_images)

  return true_pos_images, true_neg_images, false_pos_images, false_neg_images


def generate_cam(img_array, model, last_conv_layer_name, pred_index=None, respond=False):
    
    grad_model = keras.models.Model(
        [model.inputs], [model.get_layer(
            last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    last_conv_layer_output = last_conv_layer_output[0]

    if respond:
        respond_weights = np.sum(last_conv_layer_output * grads, axis=(0, 1, 2)) / \
            (np.sum(last_conv_layer_output + 1e-10, axis=(0, 1, 2)))

        heatmap = last_conv_layer_output * respond_weights
    else:
        heatmap = last_conv_layer_output * grads
    heatmap = np.sum(heatmap, axis=-1)
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()


def resize_heatmap(img, heatmap):
 
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap('jet')
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    return jet_heatmap


def visualize_side_by_side(hirescam_img, respondcam_img):
    fig = plt.figure(constrained_layout=True, figsize=(20, 10))
    axes = fig.subplots(1, 2)

    axes[0].set_title('HiResCAM')
    axes[0].imshow(hirescam_img)
    axes[1].set_title('Respond-CAM')
    axes[1].imshow(respondcam_img)
    plt.show()


def superimpose_heatmap(img, heatmap, alpha=0.4):
    superimposed_img = heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(
        superimposed_img)

    return superimposed_img


def lime_explanation(images, class_labels, model, explainer):

  def predict_function(img):
      return model.predict(img, verbose=0)
  
  lime_imgs = []
  for i, preprocessed_img in enumerate(images):
    # Generate explanations for the image
    explanation = explainer.explain_instance(preprocessed_img.astype('double'), predict_function, top_labels=1)

    # Get the top prediction and its label
    pred_idx = np.argmax(model.predict(preprocessed_img.reshape(-1,224,224,3), verbose=0),-1)
    print(pred_idx)
    top_label = class_labels[pred_idx[0]]

    # Get the explanation for the top prediction
    temp, mask = explanation.get_image_and_mask(pred_idx[0], positive_only=True, num_features=10, hide_rest=True)
    lime_img = mark_boundaries(temp / 2 + 0.5, mask)

    lime_imgs.append(lime_img)

  return lime_imgs


def shap_explaination(images, model, explainer):

  for image in images:
    shap_values = explainer(np.expand_dims(image,0), outputs=shap.Explanation.argsort.flip[:5])
    shap.image_plot(shap_values)


def plot_xia_interpretability(img, model, efficientnet_model, replace2linear, shap_explainer, lime_explainer, class_labels, layer_name, layer_id):
    
    pred = model.predict(img, verbose=0).argmax(-1)[0]
    score = CategoricalScore([pred])

    fig = plt.figure(constrained_layout=True, figsize=(14,4))
    gs = GridSpec(2, 7, figure=fig)
    ax1 = fig.add_subplot(gs[:2, :2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 3])
    ax4 = fig.add_subplot(gs[0, 4])
    ax5 = fig.add_subplot(gs[0, 5])
    ax6 = fig.add_subplot(gs[0, 6])
    ax7 = fig.add_subplot(gs[1, 2])
    ax8 = fig.add_subplot(gs[1, 3])
    ax9 = fig.add_subplot(gs[1, 4])
    ax10 = fig.add_subplot(gs[1, 5])
    ax11 = fig.add_subplot(gs[1, 6])

    # Plotting the original image
    ax1.imshow(img[0]/255)
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Plotting Saliency map
    saliency_map, smooth_grad = create_saliency_model(model, img, replace2linear, score)
    ax2.imshow(saliency_map)
    ax2.set_title('Saliency')
    ax2.axis('off')

    ax3.imshow(smooth_grad)
    ax3.set_title('SmoothGrad')
    ax3.axis('off')

    # Plotting Gradcam heatmap
    grad_cam_heatmap = create_gradcam_model(model, replace2linear, score, img, efficientnet_model, layer_id)
    ax4.imshow(img[0]/255)
    ax4.imshow(grad_cam_heatmap, cmap='jet', alpha=0.5)
    ax4.set_title('GradCam')
    ax4.axis('off')

    # Plotting Gradcam++ heatmap
    gradcam_plus_plus_heatmap = create_gradcam_plus_plus_model(model, replace2linear, score, img, efficientnet_model, layer_id)
    ax5.imshow(img[0]/255)
    ax5.imshow(gradcam_plus_plus_heatmap, cmap='jet', alpha=0.5)
    ax5.set_title('GradCam++')
    ax5.axis('off')

    # Plotting Scorecam heatmap
    scorecam_cam_heatmap = create_scorecam_model(model, efficientnet_model, score, img, layer_id)
    ax6.imshow(img[0]/255)
    ax6.imshow(scorecam_cam_heatmap, cmap='jet', alpha=0.5)
    ax6.set_title('ScoreCam')
    ax6.axis('off')

    # Plotting Faster Scorecam heatmap
    faster_scorecam_heatmap = create_faster_scorecam_model(model, replace2linear, score, img, efficientnet_model, layer_id)
    ax7.imshow(img[0]/255)
    ax7.imshow(faster_scorecam_heatmap, cmap='jet', alpha=0.5)
    ax7.set_title('Faster ScoreCam')
    ax7.axis('off')

    # Plotting Lime Explanation
    lime_img = create_lime_explanation(img, class_labels, model, lime_explainer)
    ax8.imshow(lime_img)
    ax8.set_title('Lime')
    ax8.axis('off')

    # Plotting Superimposed Hires Image
    superimposed_hires_img, superimposed_respond_img, hires_heatmap, respond_heatmap = create_heatmap_and_superimpose(img, efficientnet_model, layer_name)
    ax9.imshow(superimposed_hires_img)
    ax9.set_title('HiresCam')
    ax9.axis('off')

    # Plotting Superimposed Respond Image
    ax10.imshow(superimposed_respond_img)
    ax10.set_title('RespondCam')
    ax10.axis('off')

    methods_dict = {
        'Gradcam': {'intensity': 0.8, 'heatmap': grad_cam_heatmap},
        'Gradcam++': {'intensity': 0.7, 'heatmap': gradcam_plus_plus_heatmap},
        'Scorecam': {'intensity': 0.6, 'heatmap': scorecam_cam_heatmap},
        'Faster Scorecam': {'intensity': 0.7, 'heatmap': faster_scorecam_heatmap}
    }

    combined_heatmap = combine_heatmaps({method: (info['intensity'], info['heatmap']) for method, info in methods_dict.items()})
    ax11.imshow(combined_heatmap, cmap='jet')
    ax11.set_title('Combined Heatmap')
    ax11.axis('off')

    fig.suptitle("Model Interpreability")
    plt.tight_layout()
    plt.show()

    create_shap_explanation(img, model, shap_explainer)

    return combined_heatmap