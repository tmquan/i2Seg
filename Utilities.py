import numpy as np
import argparse
import os, glob, time, sys
import skimage.io

import PIL
from PIL import Image
import Augmentor
import malis
import malis_loss
from malis_loss import *
from natsort import natsorted
###############################################################################
import tensorflow as tf
import tensorpack.tfutils.symbolic_functions as symbf

from tensorpack import *
from tensorpack.dataflow import  *              #dataset
from tensorpack.utils import *                  #logger
from tensorpack.utils.gpu import *              #get_nr_gpu
from tensorpack.utils.utils import *            #get_rng
from tensorpack.tfutils import *                #optimizer, gradproc
from tensorpack.tfutils.summary import *        #add_moving_summary, add_param_summary, add_tensor_summary
from tensorpack.tfutils.scope_utils import *    #auto_reuse_variable_scope
###############################################################################
from tensorlayer.cost import * #binary_cross_entropy, absolute_difference_error, dice_coe, cross_entropy
###############################################################################

class ClipCallback(Callback):
    def _setup_graph(self):
        vars = tf.trainable_variables()
        ops = []
        for v in vars:
            n = v.op.name
            if not n.startswith('discrim/'):
                continue
            logger.info("Clip {}".format(n))
            ops.append(tf.assign(v, tf.clip_by_value(v, -0.01, 0.01)))
        self._op = tf.group(*ops, name='clip')

    def _trigger_step(self):
        self._op.run()

###############################################################################
# Utility function for scaling 
###############################################################################
def tf_2tanh(x, maxVal = 255.0, name='ToRangeTanh'):
    with tf.variable_scope(name):
        return (x / maxVal - 0.5) * 2.0

def tf_2imag(x, maxVal = 255.0, name='ToRangeImag'):
    with tf.variable_scope(name):
        return (x / 2.0 + 0.5) * maxVal

def np_2tanh(x, maxVal = 255.0, name='ToRangeTanh'):
    return (x / maxVal - 0.5) * 2.0

def np_2imag(x, maxVal = 255.0, name='ToRangeImag'):
    return (x / 2.0 + 0.5) * maxVal


###############################################################################
# Various types of activations
###############################################################################
def INReLU(x, name=None):
    x = InstanceNorm('inorm', x)
    return tf.nn.relu(x, name=name)

def INLReLU(x, name=None):
    x = InstanceNorm('inorm', x)
    return tf.nn.leaky_relu(x, name=name)
    
def BNLReLU(x, name=None):
    x = BatchNorm('bn', x)
    return tf.nn.leaky_relu(x, name=name)



###############################################################################
def np_seg_to_cnn(seg):
    # return the relative first voxel-encoded of the segmentation. 
    seg = np.squeeze(seg)
    seg = seg.astype(np.float32)
    dimz, dimy, dimx = seg.shape
    retz = np.zeros_like(seg) #np.zeros(seg.shape, dtype=np.float32)
    rety = np.zeros_like(seg) #np.zeros(seg.shape, dtype=np.float32)
    retx = np.zeros_like(seg) #np.zeros(seg.shape, dtype=np.float32)
    #
    # First extract the label
    labels, ind_1d = np.unique(seg, True)

    idz, idy, idx = np.unravel_index(ind_1d, (dimz, dimy, dimx))
    for label, iz, iy, ix in zip(labels, idz, idy, idx):
        retz[seg==label] = (iz+1.0)*1.0/dimz
        rety[seg==label] = (iy+1.0)*1.0/dimy
        retx[seg==label] = (ix+1.0)*1.0/dimx

    retz[seg==0] = 0.0
    rety[seg==0] = 0.0
    retx[seg==0] = 0.0
    ret = np.stack((retz, rety, retx), -1) # Make 3 channels
    return ret
###############################################################################
def np_seg_to_aff(seg, nhood=malis.mknhood3d(1)):
    # return lambda seg, nhood: malis.seg_to_affgraph (seg, nhood).astype(np.float32)
    seg = np.squeeze(seg)
    seg = seg.astype(np.int32)
    ret = malis.seg_to_affgraph(seg, nhood) # seg zyx
    ret = ret.astype(np.float32)
    ret = np.squeeze(ret) # ret 3zyx
    ret = np.transpose(ret, [1, 2, 3, 0])# ret zyx3
    return ret
def tf_seg_to_aff(seg, nhood=tf.constant(malis.mknhood3d(1)), name='SegToAff'):
    # Squeeze the segmentation to 3D
    seg = tf.cast(seg, tf.int32)
    # Define the numpy function to transform segmentation to affinity graph
    # np_func = lambda seg, nhood: malis.seg_to_affgraph (seg, nhood).astype(np.float32)
    # Convert the numpy function to tensorflow function
    tf_func = tf.py_func(np_seg_to_aff, [seg, nhood], [tf.float32], name=name)
    # Reshape the result, notice that layout format from malis is 3, dimx, dimy, dimx
    # ret = tf.reshape(tf_func[0], [3, seg.shape[0], seg.shape[1], seg.shape[2]])
    # Transpose the result so that the dimension 3 go to the last channel
    # ret = tf.transpose(ret, [1, 2, 3, 0])
    # print seg.get_shape().as_list()
    ret = tf.reshape(tf_func[0], [seg.shape[0], seg.shape[1], seg.shape[2], 3])
    # print ret.get_shape().as_list()
    return ret
###############################################################################
def np_aff_to_seg(aff, nhood=malis.mknhood3d(1), threshold=np.array([0.5]) ):
    aff = np.transpose(aff, [3, 0, 1, 2]) # zyx3 to 3zyx
    ret = malis.connected_components_affgraph((aff > threshold[0]).astype(np.int32), nhood)[0].astype(np.int32) 
    ret = skimage.measure.label(ret).astype(np.float32)
    return ret
def tf_aff_to_seg(aff, nhood=tf.constant(malis.mknhood3d(1)), threshold=tf.constant(np.array([0.5])), name='AffToSeg'):
    # Define the numpy function to transform affinity to segmentation
    # def np_func (aff, nhood, threshold):
    #   aff = np.transpose(aff, [3, 0, 1, 2]) # zyx3 to 3zyx
    #   ret = malis.connected_components_affgraph((aff > threshold[0]).astype(np.int32), nhood)[0].astype(np.int32) 
    #   ret = skimage.measure.label(ret).astype(np.int32)
    #   return ret
    # print aff.get_shape().as_list()
    # Convert numpy function to tensorflow function
    tf_func = tf.py_func(np_aff_to_seg, [aff, nhood, threshold], [tf.float32], name=name)
    ret = tf.reshape(tf_func[0], [aff.shape[0], aff.shape[1], aff.shape[2]])
    ret = tf.expand_dims(ret, axis=-1)
    # print ret.get_shape().as_list()
    return ret









'''
MIT License
Copyright (c) 2018 Fanjin Zeng
This work is licensed under the terms of the MIT license, see <https://opensource.org/licenses/MIT>.  
'''

def sliding_window_view(x, shape, step=None, subok=False, writeable=False):
    """
    Create sliding window views of the N dimensions array with the given window
    shape. Window slides across each dimension of `x` and provides subsets of `x`
    at any window position.
    Parameters
    ----------
    x : ndarray
        Array to create sliding window views.
    shape : sequence of int
        The shape of the window. Must have same length as number of input array dimensions.
    step: sequence of int, optional
        The steps of window shifts for each dimension on input array at a time.
        If given, must have same length as number of input array dimensions.
        Defaults to 1 on all dimensions.
    subok : bool, optional
        If True, then sub-classes will be passed-through, otherwise the returned
        array will be forced to be a base-class array (default).
    writeable : bool, optional
        If set to False, the returned array will always be readonly view.
        Otherwise it will return writable copies(see Notes).
    Returns
    -------
    view : ndarray
        Sliding window views (or copies) of `x`. view.shape = (x.shape - shape) // step + 1
    See also
    --------
    as_strided: Create a view into the array with the given shape and strides.
    broadcast_to: broadcast an array to a given shape.
    Notes
    -----
    ``sliding_window_view`` create sliding window views of the N dimensions array
    with the given window shape and its implementation based on ``as_strided``.
    Please note that if writeable set to False, the return is views, not copies
    of array. In this case, write operations could be unpredictable, so the return
    views is readonly. Bear in mind, return copies (writeable=True), could possibly
    take memory multiple amount of origin array, due to overlapping windows.
    For some cases, there may be more efficient approaches, such as FFT based algo discussed in #7753.
    Examples
    --------
    >>> i, j = np.ogrid[:3,:4]
    >>> x = 10*i + j
    >>> shape = (2,2)
    >>> sliding_window_view(x, shape)
    array([[[[ 0,  1],
             [10, 11]],
            [[ 1,  2],
             [11, 12]],
            [[ 2,  3],
             [12, 13]]],
           [[[10, 11],
             [20, 21]],
            [[11, 12],
             [21, 22]],
            [[12, 13],
             [22, 23]]]])
    >>> i, j = np.ogrid[:3,:4]
    >>> x = 10*i + j
    >>> shape = (2,2)
    >>> step = (1,2)
    >>> sliding_window_view(x, shape, step)
    array([[[[ 0,  1],
             [10, 11]],
            [[ 2,  3],
             [12, 13]]],
           [[[10, 11],
             [20, 21]],
            [[12, 13],
             [22, 23]]]])
    """
    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False, subok=subok)

    try:
        shape = np.array(shape, np.int)
    except:
        raise TypeError('`shape` must be a sequence of integer')
    else:
        if shape.ndim > 1:
            raise ValueError('`shape` must be one-dimensional sequence of integer')
        if len(x.shape) != len(shape):
            raise ValueError("`shape` length doesn't match with input array dimensions")
        if np.any(shape <= 0):
            raise ValueError('`shape` cannot contain non-positive value')

    if step is None:
        step = np.ones(len(x.shape), np.intp)
    else:
        try:
            step = np.array(step, np.intp)
        except:
            raise TypeError('`step` must be a sequence of integer')
        else:
            if step.ndim > 1:
                raise ValueError('`step` must be one-dimensional sequence of integer')
            if len(x.shape)!= len(step):
                raise ValueError("`step` length doesn't match with input array dimensions")
            if np.any(step <= 0):
                raise ValueError('`step` cannot contain non-positive value')

    o = (np.array(x.shape)  - shape) // step + 1 # output shape
    if np.any(o <= 0):
        raise ValueError('window shape cannot larger than input array shape')

    strides = x.strides
    view_strides = strides * step

    view_shape = np.concatenate((o, shape), axis=0)
    view_strides = np.concatenate((view_strides, strides), axis=0)
    #view = np.lib.stride_tricks.as_strided(x, view_shape, view_strides, subok=subok, writeable=writeable)
    view = np.lib.stride_tricks.as_strided(x, view_shape, view_strides, subok=subok)#, writeable=writeable)

    if writeable:
        return view.copy()
    else:
        return view