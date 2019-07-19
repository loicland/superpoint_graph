from __future__ import division
from __future__ import print_function
from builtins import range

import numpy as np

# extended official code from http://www.semantic3d.net/scripts/metric.py
class ConfusionMatrix:
  """Streaming interface to allow for any source of predictions. Initialize it, count predictions one by one, then print confusion matrix and intersection-union score"""
  def __init__(self, number_of_labels = 2):
    self.number_of_labels = number_of_labels
    self.confusion_matrix = np.zeros(shape=(self.number_of_labels,self.number_of_labels))
  def count_predicted(self, ground_truth, predicted, number_of_added_elements=1):
    self.confusion_matrix[ground_truth][predicted] += number_of_added_elements

  def count_predicted_batch(self, ground_truth_vec, predicted): # added
    for i in range(ground_truth_vec.shape[0]):
      self.confusion_matrix[:,predicted[i]] += ground_truth_vec[i,:]
     
  def count_predicted_batch_hard(self, ground_truth_vec, predicted): # added
    for i in range(ground_truth_vec.shape[0]):
      self.confusion_matrix[ground_truth_vec[i],predicted[i]] += 1

  """labels are integers from 0 to number_of_labels-1"""
  def get_count(self, ground_truth, predicted):
    return self.confusion_matrix[ground_truth][predicted]
  """returns list of lists of integers; use it as result[ground_truth][predicted]
     to know how many samples of class ground_truth were reported as class predicted"""
  def get_confusion_matrix(self):
    return self.confusion_matrix
  """returns list of 64-bit floats"""
  def get_intersection_union_per_class(self):
    matrix_diagonal = [self.confusion_matrix[i][i] for i in range(self.number_of_labels)]
    errors_summed_by_row = [0] * self.number_of_labels
    for row in range(self.number_of_labels):
      for column in range(self.number_of_labels):
        if row != column:
          errors_summed_by_row[row] += self.confusion_matrix[row][column]
    errors_summed_by_column = [0] * self.number_of_labels
    for column in range(self.number_of_labels):
      for row in range(self.number_of_labels):
        if row != column:
          errors_summed_by_column[column] += self.confusion_matrix[row][column]

    divisor = [0] * self.number_of_labels
    for i in range(self.number_of_labels):
      divisor[i] = matrix_diagonal[i] + errors_summed_by_row[i] + errors_summed_by_column[i]
      if matrix_diagonal[i] == 0:
        divisor[i] = 1

    return [float(matrix_diagonal[i]) / divisor[i] for i in range(self.number_of_labels)]
  """returns 64-bit float"""

  def get_overall_accuracy(self):
    matrix_diagonal = 0
    all_values = 0
    for row in range(self.number_of_labels):
      for column in range(self.number_of_labels):
        all_values += self.confusion_matrix[row][column]
        if row == column:
          matrix_diagonal += self.confusion_matrix[row][column]
    if all_values == 0:
      all_values = 1
    return float(matrix_diagonal) / all_values


  def get_average_intersection_union(self):
    values = self.get_intersection_union_per_class()
    class_seen = ((self.confusion_matrix.sum(1)+self.confusion_matrix.sum(0))!=0).sum()
    return sum(values) / class_seen

  def get_mean_class_accuracy(self):  # added
    re = 0
    for i in range(self.number_of_labels):
        re = re + self.confusion_matrix[i][i] / max(1,np.sum(self.confusion_matrix[i,:]))
    return re/self.number_of_labels

  def count_gt(self, ground_truth):
      return self.confusion_matrix[ground_truth,:].sum()

def compute_predicted_transitions(in_component, edg_source, edg_target):
 
  pred_transitions = in_component[edg_source] != in_component[edg_target]
  return pred_transitions

#-----------------------------------------------------------
def compute_boundary_recall(is_transition, pred_transitions):
  return 100*((is_transition==pred_transitions)*is_transition).sum()/is_transition.sum()

#-----------------------------------------------------------
def compute_boundary_precision(is_transition, pred_transitions):
  return 100*((is_transition==pred_transitions)*pred_transitions).sum()/pred_transitions.sum()
#--------------------------------------------

def mode(array, only_freq=False):
  value, counts = np.unique(array, return_counts=True)
  if only_freq:
     return np.amax(counts)
  else:
     return value[np.argmax(counts)], np.amax(counts)
#------------------------------------------------
def compute_OOA(components, labels):
  hard_labels = labels.argmax(1)
  correct_labels=0
  for comp in components:
      dump, freq = mode(hard_labels[comp])
      correct_labels+=freq
  return 100*correct_labels/len(labels)
