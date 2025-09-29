import itertools
import numpy as np
import seaborn as sns
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def compute_roc_auc(y_true, y_probs, classes):
    num_classes = len(classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    return fpr, tpr, roc_auc

def plot_roc_curve(fpr, tpr, roc_auc, classes):
    plt.figure(figsize=(10, 8))
    num_classes = len(classes)

    # Plot ROC curves
    for i in range(num_classes):
        label = f'ROC curve (class {classes[i]}) (AUC = {roc_auc[i]:.2f})'
        plt.plot(fpr[i], tpr[i], lw=2, label=label)

    plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})', linestyle='-', linewidth=4)

    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')  # Random guess line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    plt.show()

def compute_precision_recall(y_true, y_probs, classes):
    precision = dict()
    recall = dict()
    average_precision = dict()
    num_classes = len(classes)
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_probs[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_probs[:, i])

    return precision, recall, average_precision

def plot_precision_recall_curve(precision, recall, average_precision, classes, overall_precision, overall_recall, overall_average_precision):
    num_classes = len(classes)
    plt.figure(figsize=(10, 8))

    for i in range(num_classes):
        plt.plot(recall[i], precision[i], lw=2, label=f'Precision-recall curve (class {classes[i]}) (area = {average_precision[i]:.2f})')

    plt.plot(overall_recall, overall_precision, label=f'Overall Precision-recall curve (area = {overall_average_precision:.2f})', linestyle='-', linewidth=4)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()


def plot_training(hist):
    '''
    This function take training model and plot history of accuracy and losses with the best epoch in both of them.
    '''

    # Define needed variables
    tr_acc = hist.history['accuracy']
    tr_loss = hist.history['loss']
    val_acc = hist.history['val_accuracy']
    val_loss = hist.history['val_loss']
    index_loss = np.argmin(val_loss)
    val_lowest = val_loss[index_loss]
    index_acc = np.argmax(val_acc)
    acc_highest = val_acc[index_acc]
    Epochs = [i+1 for i in range(len(tr_acc))]
    loss_label = f'best epoch= {str(index_loss + 1)}'
    acc_label = f'best epoch= {str(index_acc + 1)}'

    # Plot training history
    plt.figure(figsize= (20, 8))
    plt.style.use('fivethirtyeight')

    plt.subplot(1, 2, 1)
    plt.plot(Epochs, tr_loss, 'r', label= 'Training loss')
    plt.plot(Epochs, val_loss, 'g', label= 'Validation loss')
    plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'blue', label= loss_label)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(Epochs, tr_acc, 'r', label= 'Training Accuracy')
    plt.plot(Epochs, val_acc, 'g', label= 'Validation Accuracy')
    plt.scatter(index_acc + 1 , acc_highest, s= 150, c= 'blue', label= acc_label)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout
    plt.show()


def plot_confusion_matrix(cm, classes, normalize= False, title= 'Confusion Matrix'):
	'''
	This function plot confusion matrix method from sklearn package.
	'''

	plt.figure(figsize= (4, 4))
	plt.imshow(cm, interpolation= 'nearest')
	plt.title(title)
	plt.colorbar()

	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation= 45)
	plt.yticks(tick_marks, classes)

	if normalize:
		cm = cm.astype('float') / cm.sum(axis= 1)[:, np.newaxis]
		print('Normalized Confusion Matrix')

	else:
		print('Confusion Matrix, Without Normalization')

	print(cm)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j], horizontalalignment= 'center', color= 'white' if cm[i, j] > thresh else 'black')

	plt.tight_layout()
	plt.ylabel('True Label')
	plt.xlabel('Predicted Label')


def plot_dataset_classwise_distributions(train_df, valid_df, test_df):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    train_counts = train_df.labels.value_counts()
    train_counts.plot(kind='pie', title="Training Classes", ax=axes[0], autopct=lambda p: '{:.0f}\n({:.1f}%)'.format(p * sum(train_counts) / 100, p))
    axes[0].set_xlabel('')
    axes[0].set_ylabel('')
    valid_counts = valid_df.labels.value_counts()
    valid_counts.plot(kind='pie', title="Validation Classes", ax=axes[1], autopct=lambda p: '{:.0f}\n({:.1f}%)'.format(p * sum(valid_counts) / 100, p))
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')
    test_counts = test_df.labels.value_counts()
    test_counts.plot(kind='pie', title="Testing Classes", ax=axes[2], autopct=lambda p: '{:.0f}\n({:.1f}%)'.format(p * sum(test_counts) / 100, p))
    axes[2].set_xlabel('')
    axes[2].set_ylabel('')
    plt.tight_layout()
    plt.show()


def plot_dataset_split_distribution(train_df, valid_df, test_df):
    total_images = len(train_df) + len(valid_df) + len(test_df)
    train_ratio = len(train_df) / total_images
    valid_ratio = len(valid_df) / total_images
    test_ratio = len(test_df) / total_images
    split_ratios = [train_ratio, valid_ratio, test_ratio]
    labels = ['Train', 'Valid', 'Test']
    plt.figure(figsize=(6, 6))
    plt.pie(split_ratios, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Data Split Ratio")
    plt.show()
