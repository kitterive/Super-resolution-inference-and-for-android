import os
import scipy.misc
import numpy as np

from utils import pp
import tensorflow as tf
import cv2 as cv
from scipy.misc import imresize

dataset_name = "ILSVRC2012"
CHECKPOINT_DIR = "./checkpoint"
file_dir = './predicts/'
batch_size = 64
scale_factor = 4
input_img_name = "reference.png"
#input_img_name = "g1.jpg"

def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,name="deconv2d"):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],initializer=tf.random_normal_initializer(stddev=stddev))
        
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        
        return deconv
            
def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)
        
def _phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(axis=1, num_or_size_splits=a, value=X)  # a, [bsize, b, r, r]
    X = tf.concat(axis=2, values=[tf.squeeze(x, axis=1) for x in X])  # bsize, b, a*r, r
    X = tf.split(axis=1, num_or_size_splits=b, value=X)  # b, [bsize, a*r, r]
    X = tf.concat(axis=2, values=[tf.squeeze(x, axis=1) for x in X])  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a*r, b*r, 1))


def PS(X, r, color=False):
    if color:
        Xc = tf.split(axis=3, num_or_size_splits=3, value=X)
        X = tf.concat(axis=3, values=[_phase_shift(x, r) for x in Xc])
    else:
        X = _phase_shift(X, r)
    return X
            
def predict_generator(test_img, input_size):
    # project `z` and reshape
    test_img_height = input_size[0]
    test_img_width = input_size[1]
    h0 = deconv2d(test_img, [1, test_img_height, test_img_width, 64], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h0')
    #h0 = lrelu(h0)
    h0 = tf.nn.relu(h0)

    h1 = deconv2d(h0, [1, test_img_height, test_img_width, 64],  d_h=1, d_w=1, name='g_h1')
    #h1 = lrelu(h1)
    h1 = tf.nn.relu(h1)

    h2 = deconv2d(h1, [1, test_img_height, test_img_width, 3*scale_factor*scale_factor], d_h=1, d_w=1, name='g_h2')
    h2 = PS(h2, scale_factor, color=True)

    return tf.nn.tanh(h2, name='tanh_out')

 
def main(_):
    file_name = input_img_name
    file_name_main_str = file_name.split('.')[0]
    test_img = cv.imread(file_dir + file_name)
    up_size_rows = test_img.shape[0]
    up_size_cols = test_img.shape[1]
    print "up_size_rows=" +str(up_size_rows) +   "    up_size_cols=" + str(up_size_cols)
    
    down_size_rows = up_size_rows // scale_factor
    down_size_cols = up_size_cols // scale_factor
    print "down_size_rows=" +str(down_size_rows) +   "    down_size_cols=" + str(down_size_cols)
    
 
    input_img = scipy.misc.imread(file_dir+input_img_name).astype(np.float)
    
    sample_input = cv.resize(input_img,(down_size_cols,down_size_rows), interpolation=cv.INTER_CUBIC)
    predict_input_batch = np.array(sample_input).astype(np.float32)
    
    #print predict_input_batch
 
    print "predict_input_batch shape is ", predict_input_batch.shape

    #print predict_input_batch
    
    up_res_img = cv.resize(sample_input, (up_size_cols, up_size_rows), interpolation=cv.INTER_CUBIC)
    
    sample_input_image = np.array([sample_input]).astype(np.float32)
    up_res_image = np.array([up_res_img]).astype(np.float32)
    print "sample_input_image shape is: ",sample_input_image.shape
    print "up_res_image shape is:", up_res_image.shape
    
    scipy.misc.imsave(file_dir+file_name_main_str+'_small.png',sample_input_image[0])
    scipy.misc.imsave(file_dir+file_name_main_str+'_cubic.png',up_res_image[0])
         
    input_placeholder = tf.placeholder(tf.float32, [1, down_size_rows, down_size_cols, 3], name='predict_input')
         
    img_size = (down_size_rows, down_size_cols)
    
    model_dir = "%s_%s" % (dataset_name, batch_size)
    checkpoint_dir = os.path.join(CHECKPOINT_DIR, model_dir)
    print(" [*] Reading checkpoints...")
     
    out_img = predict_generator(input_placeholder, img_size)
     
    sess = tf.Session() 
    
    # we still need to initialize all variables even when we use Saver's restore method.
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    #saver = tf.train.Saver()
  
    print " [*] begin to load...."
    
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        
        print "load secceed! mode is :" ,os.path.join(checkpoint_dir, ckpt_name)
        
    else:
        print "load failed!"
      
    out_highres_img = sess.run(out_img, feed_dict={input_placeholder:[predict_input_batch]})
    print "out_highres_img shape is: ", out_highres_img.shape
    
    tf.train.write_graph(sess.graph_def, 'models/', 'graph.pbtxt')
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['tanh_out'])
    with tf.gfile.FastGFile("models/graph_valuble.pb", mode = 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    
   
    scipy.misc.imsave(file_dir+file_name_main_str+'_out.png', out_highres_img[0])    
        
if __name__ == '__main__':
    tf.app.run()
    
           
