import tensorflow as tf
import numpy as np
import pandas as pd
import functools
import random

def make_csv_dataset(file_path, label_name, batch_size, 
                     numeric_features, categorical_features, 
                     normalize_numerical_features=True, 
                     shuffle=True):
  """
    Load a dataset from a csv file.
    
    label_name should be None for a testset
    numeric_features = ['Survived, 'Age' ...]
    categorical_features = { 'Sex': ['male', 'female']}
  """
  normalizer = None
  if normalize_numerical_features:
    normalizer = _make_normalizer(file_path, numeric_features)
  
  dataset, preprocessing_layer = _make_dataset(file_path, label_name, 
      batch_size, numeric_features, categorical_features, normalizer, shuffle)
  
  #dataset = packed_train_data.shuffle(4000)
  return dataset, preprocessing_layer
    
def print_dataset(dataset, max_elems=100):
  """
  Prints first 'max_elems' elements of a dataset
  """
  count = 0
  for i in dataset:
    print(i)
    count = count + 1
    if count > max_elems:
      break
  
def _normalize_numeric_data(data, mean, std):
  # Center the data
  return (data-mean)/std

def _make_normalizer(train_file_path, numeric_features):
  desc = pd.read_csv(train_file_path, na_values='?')[numeric_features].describe()
  MEAN = np.array(desc.T['mean'])
  STD = np.array(desc.T['std'])
  normalizer = functools.partial(_normalize_numeric_data, mean=MEAN, std=STD)
  return normalizer


def _make_dataset(file_path, label_name, batch_size, numeric_features, categorical_features, normalizer, shuffle):
  select_columns = []
  if numeric_features:
    select_columns += numeric_features
  if categorical_features:
    select_columns += list(categorical_features.keys())
  if label_name:
    select_columns += [label_name]
  raw_train_data = _make_csv_dataset(file_path, label_name, batch_size, select_columns, shuffle=shuffle)
  
  #dataset = dataset.apply(tf.data.experimental.unique())
  
  # Pack numeric cols together so we can treat them all the same.
  packed_train_data = raw_train_data.map(PackNumericFeatures(numeric_features))
  numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(numeric_features)])
  columns = [numeric_column]
  
  if categorical_features:
    categorical_columns = []
    for feature, vocab in categorical_features.items():
      cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
      categorical_columns.append(tf.feature_column.indicator_column(cat_col))
    columns += categorical_columns
  
  preprocessing_layer = tf.keras.layers.DenseFeatures(columns)
 
  return packed_train_data, preprocessing_layer


def remove_duplicates(file_path, cols=None):
  df = pd.read_csv(file_path, na_values='?')#[numeric_features].describe()
  if cols:
    df = df[cols]
  simplified = df.drop_duplicates()
  original_len = len(df)
  new_len =len(simplified)
  print(original_len - new_len, "dupes removed out of", original_len)
  print((original_len - new_len) / original_len * 100, "% dupes")
  filename = file_path[:-4] + "_nodups.csv"
  simplified.to_csv(filename)
  print("New csv outputted to:", filename)

def _remove_duplicates_tf(dataset, print_summary=False):
  print(dataset)
  if print_summary:
    original_size = len(list(dataset))
  
  dataset = dataset.apply(tf.data.experimental.unique())
  
  if print_summary:
    print("Original", original_size, "Unique", len(list(dataset)))

  

def _make_csv_dataset(file_path, label_name, batch_size, 
                     select_columns=None, shuffle=True, **kwargs):
  """
      select_columns array of column names
  """

  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=batch_size,
      label_name=label_name,
      na_value="?",
      num_epochs=1,
      ignore_errors=True, 
      select_columns=select_columns,
      shuffle=shuffle,
      **kwargs)
  return dataset

class PackNumericFeatures(object):
  def __init__(self, names, new_name='numeric'):
    self.names = names
    self.new_name = new_name

  def __call__(self, features, labels=None):
    numeric_features = [features.pop(name) for name in self.names]
    numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
    numeric_features = tf.stack(numeric_features, axis=-1)
    features[self.new_name] = numeric_features
    return features, labels
    

def _show_batch(dataset):
  for batch, _ in dataset.take(2):
    for key, value in batch.items():
      print("{:20s}: {}".format(key,value.numpy()))
      

def shuffle_file(file_path):
  lines = open(file_path).readlines()
  header = lines[0]
  del lines[0]
  random.shuffle(lines)
  lines.insert(0, header)
  filename = file_path[:-4] + "_shuffled.csv"
  open(filename, 'w').writelines(lines)

def _first_time_dataset_preprocess(file_path, cols):
  """
    Call the first time you use a new dataset
  """
  remove_duplicates(file_path, cols)
  
