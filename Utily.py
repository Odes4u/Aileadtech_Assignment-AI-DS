import seaborn as sns
import matplotlib.pyplot as plt


# function to create labeled barplots

def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  # length of the column
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 1, 4))
    else:
        plt.figure(figsize=(n + 1, 4))

    plt.xticks(rotation=45, fontsize=8)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=8,
            xytext=(0, 5),
            textcoords="offset points",
        ) # annotate the percentage

    plt.show()  # show the plot


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g pd.read_csv)
import matplotlib.pyplot as plt # matplotlib.pyplot plots data
import seaborn as sns
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score, confusion_matrix)
    # function to compute different metrics to check performance of a regression model
def model_performance_classification(model, predictors, target, threshold = 0.5):
    """
    Function to compute different metrics to check regression model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    """

    # predicting using the independent variables
    pred_proba = model.predict_proba(predictors)[:, 1]
    #cover the probability to class
    pred_class = np.round(pred_proba > threshold)
    acc = accuracy_score(target, pred_class) # to compute accuracy
    recall = recall_score(target, pred_class) # to compute recall
    precision = precision_score(target, pred_class) # to compute precision
    f1 = f1_score(target, pred_class)

    # creating a dataframe of metrics
    kf_perf = pd.DataFrame(
        {
            "Accuracy": acc,
            "Recall": recall,
            "Precision": precision,
            "F1_score": f1,
        },
        index=[0])
    conf = confusion_matrix(target, pred_class)
    plt.figure(figsize=(8, 5))
    sns.heatmap(conf, annot = True, fmt="g")
    plt.xlabel("predicted label")
    plt.ylabel("Actual label")
    plt.show()
    
    return kf_perf


    # Function to draw the feature importance diagram
def draw_importance(model, predictors):
    feature_names = predictors.columns.to_list() #get the feature names
    importances = model.feature_importances_  # get the feature importance
    indices = np.argsort(importances)   # sort the feature importance

    plt.figure(figsize = (10, 10))
    plt.title("Feature Importances")
    plt.barh(range(len(indices)), importances[indices], color = "violet", 
             align = "center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.show()