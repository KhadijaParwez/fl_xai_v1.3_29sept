from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, classification_report
import numpy as np
from plotting_utils import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, compute_roc_auc, compute_precision_recall
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, matthews_corrcoef, cohen_kappa_score
from sklearn.preprocessing import OneHotEncoder
from itertools import cycle


def test_scores(test_gen, model):
  
  cm = []

  test_labels = test_gen.labels
  
  num_classes = len(np.unique(test_labels))
  test_pred_scores = model.predict(test_gen, batch_size=1, verbose=1)
  test_preds = test_pred_scores.argmax(-1)
  test_accuracy = accuracy_score(test_labels, test_preds)

  g_dict = test_gen.class_indices
  classes = list(g_dict.keys())

  # Confusion matrix
  cm = confusion_matrix(test_labels, test_preds)
  plot_confusion_matrix(cm= cm, classes= classes, title = 'Confusion Matrix')

  # Classification report
  print(classification_report(test_labels, test_preds, target_names= classes))  
  print("Test Accuracy:", test_accuracy)
  print("Test MCC:", matthews_corrcoef(test_labels, test_preds))
  print("Test Kappa:", cohen_kappa_score(test_labels, test_preds))
  
  if num_classes == 2:

    print("Test Precision:", precision_score(test_labels, test_preds))
    print("Test Recall:", recall_score(test_labels, test_preds))
    print("Test F1 Score:", f1_score(test_labels, test_preds))
    print("Test AUC:", roc_auc_score(test_labels, test_preds))
    print("Test mAP:", average_precision_score(test_labels, test_preds))
    
  elif num_classes > 2:
  
    print("Test Precision (Weighted):", precision_score(test_labels, test_preds, average='weighted'))
    print("Test Recall (Weighted):", recall_score(test_labels, test_preds, average='weighted'))
    print("Test F1 Score (Weighted):", f1_score(test_labels, test_preds, average='weighted'))
    print("Test Precision (Micro):", precision_score(test_labels, test_preds, average='micro'))
    print("Test Recall (Micro):", recall_score(test_labels, test_preds, average='micro'))
    print("Test F1 Score (Micro):", f1_score(test_labels, test_preds, average='micro'))
    print("Test Precision (Macro):", precision_score(test_labels, test_preds, average='macro'))
    print("Test Recall (Macro):", recall_score(test_labels, test_preds, average='macro'))
    print("Test F1 Score (Macro):", f1_score(test_labels, test_preds, average='macro'))


  encoder = OneHotEncoder(sparse=False)
  y_true_onehot = encoder.fit_transform(np.array(test_labels).reshape(-1, 1))


  fpr, tpr, roc_auc = compute_roc_auc(y_true_onehot, test_pred_scores, classes)
  plot_roc_curve(fpr, tpr, roc_auc, classes)


  overall_precision, overall_recall, _ = precision_recall_curve(y_true_onehot.ravel(), test_pred_scores.ravel())
  overall_average_precision = average_precision_score(y_true_onehot.ravel(), test_pred_scores.ravel())
  precision, recall, average_precision = compute_precision_recall(y_true_onehot, test_pred_scores, classes)
  plot_precision_recall_curve(precision, recall, average_precision, classes, overall_precision, overall_recall, overall_average_precision)

  return cm
