# -*- coding: utf-8 -*-
"""
Created on Wed May  1 22:41:45 2019

@author: silen
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import tensorflow as tf
import cifar10
import pickle

def merge_dicts(*dict_args):
     """
     Given any number of dicts, shallow copy and merge into a new dict,
     precedence goes to key value pairs in latter dicts.
     """
     result = {}
     for dictionary in dict_args:     
         result.update(dictionary)
     return result


def load(filename):
    with open(filename, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
#        data = pickle.load(fo, encoding='bytes')
#        data = pickle.load(fo, encoding='iso-8859-1')
    return data

    
def get_data_sample(data_size):      
    for j in range(5):

        i=j+1   
        print (i)

        data_name = 'data_batch_%i' % i
        p=load(data_name)
        data_sample = {}        
        data_sample["batch_label"]=p["batch_label"]
        data_sample["data"]=p["data"][0:data_size,:]
        data_sample["filenames"]=p["filenames"][0:data_size]
        data_sample["labels"]=p["labels"][0:data_size]
        
        
#        arr2=p["data"][0:data_size,:]
#        list3=p["filenames"][0:data_size]
#        label=p["labels"][0:data_size]        
#        data={}
#        data['batch_label'.encode('utf-8')]='training batch 1 of %i'.encode('utf-8') % i 
#        data.setdefault('labels'.encode('utf-8'),label)
#        data.setdefault('data'.encode('utf-8'),arr2)
#        data.setdefault('filenames'.encode('utf-8'),list3)
#        output = open('sample_data_batch_%i' % i, 'wb')
#        pickle.dump(data, output)
#        output.close()

        
        output = open('sample_data_batch_%i' % i, 'wb')
        pickle.dump(data_sample, output)
        output.close()


def get_test_data_sample(data_size):      

    data_name = 'test_batch'
    p=load(data_name)
    data_sample = {}        
    data_sample["batch_label"]=p["batch_label"]
    data_sample["data"]=p["data"][0:data_size,:]
    data_sample["filenames"]=p["filenames"][0:data_size]
    data_sample["labels"]=p["labels"][0:data_size]

        
    output = open('sample_test_batch', 'wb')
    pickle.dump(data_sample, output)
    output.close()


#数据的采样比例
data_size=1000
get_data_sample(data_size)
get_test_data_sample(data_size)

#data_name = 'sample_data_batch_1.bin'
#sample_data_batch_1=load(data_name)
#
#data_name = 'data_batch_1'
#data_batch_1=load(data_name)

#FLAGS = tf.app.flags.FLAGS
#
#tf.app.flags.DEFINE_string('train_dir', './log',
#                           """Directory where to write event logs """
#                           """and checkpoint.""")
#tf.app.flags.DEFINE_integer('max_steps', 1000,
#                            """Number of batches to run.""")
#tf.app.flags.DEFINE_boolean('log_device_placement', False,
#                            """Whether to log device placement.""")
#tf.app.flags.DEFINE_integer('log_frequency', 10,
#                            """How often to log results to the console.""")
#
#
#
#
#"""Train CIFAR-10 for a number of steps."""
#with tf.Graph().as_default():
#    global_step = tf.contrib.framework.get_or_create_global_step()
#    
#    with tf.device('/cpu:0'):        
#        # Get images and labels for CIFAR-10.
#        images, labels = cifar10.distorted_inputs()
#    
#    # Build a Graph that computes the logits predictions from the
#    # inference model.
#    logits = cifar10.inference(images)
#
#    # Calculate loss.
#    loss = cifar10.loss(logits, labels)
#
#    # Build a Graph that trains the model with one batch of examples and
#    # updates the model parameters.
#    train_op = cifar10.train(loss, global_step)
#
#    class _LoggerHook(tf.train.SessionRunHook):
#      """Logs loss and runtime."""
#
#      def begin(self):
#        self._step = -1
#        self._start_time = time.time()
#
#      def before_run(self, run_context):
#        self._step += 1
#        return tf.train.SessionRunArgs(loss)  # Asks for loss value.
#
#      def after_run(self, run_context, run_values):
#        if self._step % FLAGS.log_frequency == 0:
#          current_time = time.time()
#          duration = current_time - self._start_time
#          self._start_time = current_time
#
#          loss_value = run_values.results
#          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
#          sec_per_batch = float(duration / FLAGS.log_frequency)
#
#          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
#                        'sec/batch)')
#          print(format_str % (datetime.now(), self._step, loss_value,
#                              examples_per_sec, sec_per_batch))
#
#    with tf.train.MonitoredTrainingSession(
#        checkpoint_dir=FLAGS.train_dir,
#        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
#               tf.train.NanTensorHook(loss),
#               _LoggerHook()],
#        config=tf.ConfigProto(
#            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
#      while not mon_sess.should_stop():
#        mon_sess.run(train_op)






