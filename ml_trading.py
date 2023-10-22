import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import xgboost
from sklearn.model_selection import train_test_split

# train and test datasets
train_data = pd.read_csv("aapl_5m_train.csv")
test_data = pd.read_csv("aapl_5m_validation.csv")