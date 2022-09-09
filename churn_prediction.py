import pandas as pd
import numpy as np
import seaborn as sns
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
df= pd.read_csv('preProcessedFile.csv')


#import sklearn methods
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import MinMaxScaler

y = df.loc[:, 'Churn'].values # select all the rows in Churn Columns and get the value
X = df.drop('Churn', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y)
cv = StratifiedShuffleSplit(n_splits=12, random_state = 12) 
hold_out = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state = 25)


scaler = MinMaxScaler() 
scaler.fit(X_train, X_test)

updated_X_train_arr = scaler.transform(X_train)
updated_X_train_df  = pd.DataFrame(updated_X_train_arr, columns=list(X.columns))
updated_X_test_arr  = scaler.transform(X_test)
updated_X_test_df   = pd.DataFrame(updated_X_test_arr, columns=list(X.columns))


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
   BackgroundLightGreen   = "\033[102m"
   BackgroundLightYellow  = "\033[103m"
   BackgroundLightBlue    = "\033[104m"
   BackgroundLightMagenta = "\033[105m"

print(color.BOLD + 'Successfully Run...... !' + color.END)

def get_best_parameters(grid):
  print(color.BOLD+"The Best Parameters Are: %s" % (grid.best_params_))

# display test scores and return result string and indexes of false samples
def get_results(test, prediction):
    str_out = "\n"
    str_out += (color.BackgroundLightGreen + 'SCORES DETAILS' + color.END)
    str_out += ("\n \n")

    #print accuracy
    accuracy = accuracy_score(test, prediction)
    str_out += (color.GREEN + "ACCURACY: {:.2f}\n".format(accuracy) + color.END)

    #print AUC score
    auc = roc_auc_score(test, prediction)
    str_out += (color.PURPLE + "AUC: {:.4f}\n".format(auc) + color.END)
    str_out += ("\n")

    #print confusion matrix
    conf_mat = confusion_matrix(test, prediction)
    cm_matrix = pd.DataFrame(conf_mat, index=[ 'Actual Negative','Actual Positive'], 
                                 columns=[ 'Predict Negative','Predict Positive'])
    sns.heatmap(cm_matrix, annot = True, fmt='d', cmap='BrBG')


#print classification report
    str_out += ("{}".format(classification_report(test, prediction)))
    str_out += (color.UNDERLINE+ "\n \nCONFUSION MATRIX:\n" +color.END)
    
    false_indexes = np.where(test != prediction)
    return str_out, false_indexes



# Logistic regression
lr = LogisticRegression(random_state = 50, max_iter=1000, tol=0.1)

# parameters 
lr_param_grid = [ {'C' : [10, 100, 300, 500, 800] } ]


# grid search for parameters
grid_1 = GridSearchCV(estimator=lr, param_grid=lr_param_grid, cv=cv)
grid_1.fit(updated_X_train_df, y_train)

get_best_parameters(grid_1)

y_pred = grid_1.predict(updated_X_test_df)
results, false = get_results(y_test, y_pred)

print(results)

"""**Logistic Regression (Holdout)**"""

grid_1_h = GridSearchCV(estimator=lr, param_grid=lr_param_grid, cv=hold_out)
grid_1_h.fit(updated_X_train_df, y_train)

get_best_parameters(grid_1_h)

# prediction results
y_pred = grid_1_h.predict(updated_X_test_df)

results, false = get_results(y_test, y_pred)

print(results)

