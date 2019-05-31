import numpy as np
import tensorflow as tf
from params import *

input_pl = tf.placeholder(tf.float32,shape=(None,None),name='input_t')

class popRNN:
    def __enter__(self):
        return self

    def __init__(self,EVOL_P):
