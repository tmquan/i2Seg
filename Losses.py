import tensorflow as tf

def tf_norm(inputs, axis=1, epsilon=1e-7,  name='safe_norm'):
    squared_norm    = tf.reduce_sum(tf.square(inputs), axis=axis, keepdims=True)
    safe_norm       = tf.sqrt(squared_norm+epsilon)
    return tf.identity(safe_norm, name=name)

def discriminative_loss_single(prediction, correct_label, feature_dim, label_shape, 
                            delta_v, delta_d, param_var, param_dist, param_reg):
    
    ''' Discriminative loss for a single prediction/label pair.
    :param prediction: inference of network
    :param correct_label: instance label
    :feature_dim: feature dimension of prediction
    :param label_shape: shape of label
    :param delta_v: cutoff variance distance
    :param delta_d: curoff cluster distance
    :param param_var: weight for intra cluster variance
    :param param_dist: weight for inter cluster distances
    :param param_reg: weight regularization
    '''

    ### Reshape so pixels are aligned along a vector
    correct_label = tf.reshape(correct_label, [label_shape[2]*label_shape[1]*label_shape[0]])
    reshaped_pred = tf.reshape(prediction, [label_shape[2]*label_shape[1]*label_shape[0], feature_dim])

    ### Count instances
    unique_labels, unique_id, counts = tf.unique_with_counts(correct_label)
    counts = tf.cast(counts, tf.float32)
    num_instances = tf.size(unique_labels)

    segmented_sum = tf.unsorted_segment_sum(reshaped_pred, unique_id, num_instances)

    mu = tf.divide(segmented_sum, tf.reshape(counts, (-1, 1)))
    mu_expand = tf.gather(mu, unique_id)

    ### Calculate l_var
    distance = tf_norm(tf.subtract(mu_expand, reshaped_pred), axis=1)
    distance = tf.subtract(distance, delta_v)
    distance = tf.clip_by_value(distance, 0., distance)
    distance = tf.square(distance)

    l_var = tf.unsorted_segment_sum(distance, unique_id, num_instances)
    l_var = tf.divide(l_var, counts)
    l_var = tf.reduce_sum(l_var)
    l_var = tf.divide(l_var, tf.cast(num_instances, tf.float32))
    
    ### Calculate l_dist
    
    # Get distance for each pair of clusters like this:
    #   mu_1 - mu_1
    #   mu_2 - mu_1
    #   mu_3 - mu_1
    #   mu_1 - mu_2
    #   mu_2 - mu_2
    #   mu_3 - mu_2
    #   mu_1 - mu_3
    #   mu_2 - mu_3
    #   mu_3 - mu_3

    mu_interleaved_rep = tf.tile(mu, [num_instances, 1])
    mu_band_rep = tf.tile(mu, [1, num_instances])
    mu_band_rep = tf.reshape(mu_band_rep, (num_instances*num_instances, feature_dim))

    mu_diff = tf.subtract(mu_band_rep, mu_interleaved_rep)
    
    # Filter out zeros from same cluster subtraction
    intermediate_tensor = tf.reduce_sum(tf.abs(mu_diff),axis=1)
    zero_vector = tf.zeros(1, dtype=tf.float32)
    bool_mask = tf.not_equal(intermediate_tensor, zero_vector)
    mu_diff_bool = tf.boolean_mask(mu_diff, bool_mask)

    mu_norm = tf_norm(mu_diff_bool, axis=1)
    mu_norm = tf.subtract(2.*delta_d, mu_norm)
    mu_norm = tf.clip_by_value(mu_norm, 0., mu_norm)
    mu_norm = tf.square(mu_norm)

    l_dist = tf.reduce_mean(mu_norm)

    ### Calculate l_reg
    l_reg = tf.reduce_mean(tf_norm(mu, axis=1))

    param_scale = 1.
    l_var = param_var * l_var
    l_dist = param_dist * l_dist
    l_reg = param_reg * l_reg

    l_var = tf.where(tf.is_nan(l_var), tf.ones_like(l_var) * 1e+2, l_var);
    l_dist = tf.where(tf.is_nan(l_dist), tf.ones_like(l_dist) * 1e+2, l_dist);
    l_reg = tf.where(tf.is_nan(l_reg), tf.ones_like(l_reg) * 1e+2, l_reg);
    loss = param_scale*(l_var + l_dist + l_reg)
    
    return loss, l_var, l_dist, l_reg


def discriminative_loss(prediction, correct_label, feature_dim, image_shape, 
                delta_v, delta_d, param_var, param_dist, param_reg):
    ''' Iterate over a batch of prediction/label and cumulate loss
    :return: discriminative loss and its three components
    '''
    def cond(label, batch, out_loss, out_var, out_dist, out_reg, i):
        return tf.less(i, tf.shape(batch)[0])

    def body(label, batch, out_loss, out_var, out_dist, out_reg, i):
        disc_loss, l_var, l_dist, l_reg = discriminative_loss_single(prediction[i], correct_label[i], feature_dim, image_shape, 
                        delta_v, delta_d, param_var, param_dist, param_reg)

        out_loss = out_loss.write(i, disc_loss)
        out_var = out_var.write(i, l_var)
        out_dist = out_dist.write(i, l_dist)
        out_reg = out_reg.write(i, l_reg)

        return label, batch, out_loss, out_var, out_dist, out_reg, i + 1

    # TensorArray is a data structure that support dynamic writing
    output_ta_loss = tf.TensorArray(dtype=tf.float32,
                   size=0,
                   dynamic_size=True)
    output_ta_var = tf.TensorArray(dtype=tf.float32,
                   size=0,
                   dynamic_size=True)
    output_ta_dist = tf.TensorArray(dtype=tf.float32,
                   size=0,
                   dynamic_size=True)
    output_ta_reg = tf.TensorArray(dtype=tf.float32,
                   size=0,
                   dynamic_size=True)

    _, _, out_loss_op, out_var_op, out_dist_op, out_reg_op, _  = tf.while_loop(cond, body, [correct_label, 
                                                        prediction, 
                                                        output_ta_loss, 
                                                        output_ta_var, 
                                                        output_ta_dist, 
                                                        output_ta_reg, 
                                                        0])
    out_loss_op = out_loss_op.stack()
    out_var_op = out_var_op.stack()
    out_dist_op = out_dist_op.stack()
    out_reg_op = out_reg_op.stack()
    
    disc_loss = tf.reduce_mean(out_loss_op)
    l_var = tf.reduce_mean(out_var_op)
    l_dist = tf.reduce_mean(out_dist_op)
    l_reg = tf.reduce_mean(out_reg_op)

    return disc_loss, l_var, l_dist, l_reg





def supervised_clustering_loss(prediction, correct_label, feature_dim, label_shape):
    Y = tf.reshape(correct_label, [label_shape[1]*label_shape[0], 1])
    F = tf.reshape(prediction, [label_shape[1]*label_shape[0], feature_dim])
    diagy = tf.reduce_sum(Y,0)
    onesy = tf.ones(diagy.get_shape())
    J = tf.matmul(Y,tf.diag(tf.rsqrt(tf.where(tf.greater_equal(diagy,onesy),diagy,onesy))))
    [S,U,V] = tf.svd(F)
    Slength = tf.cast(tf.reduce_max(S.get_shape()), tf.float32)
    maxS = tf.fill(tf.shape(S),tf.scalar_mul(tf.scalar_mul(1e-15,tf.reduce_max(S)),Slength))
    ST = tf.where(tf.greater_equal(S,maxS),tf.divide(tf.ones(S.get_shape()),S),tf.zeros(S.get_shape()))
    pinvF = tf.transpose(tf.matmul(U,tf.matmul(tf.diag(ST),V,False,True)))
    FJ = tf.matmul(pinvF,J)
    G = tf.matmul(tf.subtract(tf.matmul(F,FJ),J),FJ,False,True)
    loss = tf.reduce_sum(tf.multiply(tf.stop_gradient(G),F))
    return loss


def loss_tv_l1(image, is_L1=True, name='loss_tv_l1'):
    from tensorflow.python.framework import ops
    from tensorflow.python.ops import math_ops
    with ops.name_scope(name, 'loss_tv_l1'):
        ndims = image.get_shape().ndims
        Gx = tf.zeros_like(image)
        Gy = tf.zeros_like(image)
        Gz = tf.zeros_like(image)

        if ndims == 3:
            pass
        elif ndims == 4:
            # The input is a batch of image with shape:
            # [batch, height, width, channels].

            # Calculate the difference of neighboring pixel-values.
            # The image are shifted one pixel along the height and width by slicing.
            padded_img1 = tf.pad(image, paddings=[[0,0], [1,1], [0,0], [0,0]], mode="REFLECT")
            padded_img2 = tf.pad(image, paddings=[[0,0], [0,0], [1,1], [0,0]], mode="REFLECT")
            padded_img3 = tf.pad(image, paddings=[[1,1], [0,0], [0,0], [0,0]], mode="REFLECT")
            
            Gx = 0.5*(padded_img1[:,:-2,:,:] - padded_img1[:,2:,:,:])
            Gy = 0.5*(padded_img2[:,:,:-2,:] - padded_img2[:,:,2:,:])
            Gz = 0.5*(padded_img3[:-2,:,:,:] - padded_img3[2:,:,:,:])
            # grad = tf.sqrt(tf.add_n([tf.pow(Gx,2),tf.pow(Gy,2),tf.pow(Gz,2)]))
            # return grad
        else:
            raise ValueError('\'image\' must be either 3 or 4-dimensional.')

        if is_L1:
            grad = tf.add_n([tf.abs(Gx), tf.abs(Gy), tf.abs(Gz)])# okay
        else: # is_L2
            grad = tf.sqrt(tf.add_n([tf.square(Gx), tf.square(Gy), tf.square(Gz)])) # okay
        return tf.reduce_sum(grad, name=name)
