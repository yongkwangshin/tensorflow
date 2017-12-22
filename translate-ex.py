# translate-ex.py: illustration of RNN input encoding
# written by Sung Kyu Lim
# limsk@ece.gatech.edu
# 12/21/2017

# B1: import related packages
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# B2: 41 characters used in the dictionary
# S, E, and P are special characters used in RNN
char_arr = [c for c in 
           'SEPabcdefghijklmnopqrstuvwxyz단어나무놀이소녀키스사랑']
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

# B3: print array contents
print('char_arr:\n', char_arr)
print('num_dic:\n', num_dic)
print('number of letters in our alphabet:', dic_len, '\n')

# B4: sample input
# the english word should be of length 4, and the korean 2
# this should be a 2D array tensor
seq_data = [['wood', '나무'], ['love', '사랑']]

# B5: one-hot encoding function
def make_batch(seq_data):
    input_batch = []
    output_batch = []
    target_batch = []

    for seq in seq_data:
        # index encoding of each letter in the english word
        input = [num_dic[n] for n in seq[0]]

        # index encoding of each letter in the korean word
        # we add 'S' in the beginning as a separator.
        output = [num_dic[n] for n in ('S' + seq[1])]

        # index encoding of each letter in the korean word
        # this time we add 'E' at the end as a terminator.
        target = [num_dic[n] for n in (seq[1] + 'E')]

        # print the result
        print('word = ', seq)
        print('input:', input)
        print('output:', output)
        print('target:', target)

        # turn the index encoding to one-hot for 'input' &
        # 'output‘ using np.eye function.
        # we do not do one-hot encoding for 'target'
        # because we will use sparse_softmax optimizer.       
        input_one = np.eye(dic_len)[input]
        output_one = np.eye(dic_len)[output]

        # print the result
        print('input_one\n', input_one)
        print('output_one\n', output_one)
        print()

        # add the final results to the batch
        input_batch.append(input_one)
        output_batch.append(output_one)
        target_batch.append(target)

    # print the final result
    print('=== final result ===')
    print('input_batch:\n', input_batch)
    print('output_batch:\n', output_batch)
    print('target_batch:\n', target_batch)

    return input_batch, output_batch, target_batch

# B6: session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
input_batch, output_batch, target_batch = make_batch(seq_data)

