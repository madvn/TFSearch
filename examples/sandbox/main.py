import tensorflow as tf
import numpy as np
from constTestClass import *

def const_test_main():
    sess = tf.Session()
    const_test = ConstTest(np.ones(5)*1.)
    for i in range(5):
        print sess.run(const_test.output_T,feed_dict={input_pl:np.ones(5)*i})

if __name__ == "__main__":
    const_test_main()
