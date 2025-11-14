import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def calc_confusion_matrix(y_test, y_pred_tuned, class_names, title, prefix='all'):
    
    cm = confusion_matrix(y_test, y_pred_tuned, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, 
                annot=True, 
                fmt='.3f', 
                cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    save_file = title.replace(' ','_')
    plt.savefig(f'fig_{prefix}/{save_file}.png', dpi=300)
    plt.show()