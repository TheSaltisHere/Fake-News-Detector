import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TdifVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
