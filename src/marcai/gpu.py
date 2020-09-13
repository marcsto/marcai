import tensorflow as tf
import os

def is_gpu_available():
  return tf.test.is_gpu_available()

def disable_gpu():
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    print(is_gpu_available())