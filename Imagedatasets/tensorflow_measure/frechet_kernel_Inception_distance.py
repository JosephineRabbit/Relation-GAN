import tensorflow as tf
import functools
import numpy as np
import time
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import control_flow_ops
from scipy import misc

tfgan = tf.contrib.gan

session = tf.InteractiveSession()

def _symmetric_matrix_square_root(mat, eps=1e-10):

  s, u, v = linalg_ops.svd(mat)

  si = array_ops.where(math_ops.less(s, eps), s, math_ops.sqrt(s))

  return math_ops.matmul(
      math_ops.matmul(u, array_ops.diag(si)), v, transpose_b=True)

def trace_sqrt_product(sigma, sigma_v):

  sqrt_sigma = _symmetric_matrix_square_root(sigma)

  sqrt_a_sigmav_a = math_ops.matmul(sqrt_sigma,
                                    math_ops.matmul(sigma_v, sqrt_sigma))

  return math_ops.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))

def frechet_classifier_distance_from_activations(real_activations,
                                                 generated_activations):

    real_activations.shape.assert_has_rank(2)
    generated_activations.shape.assert_has_rank(2)

    activations_dtype = real_activations.dtype
    if activations_dtype != dtypes.float64:
        real_activations = math_ops.to_double(real_activations)
        generated_activations = math_ops.to_double(generated_activations)

    m = math_ops.reduce_mean(real_activations, 0)
    m_w = math_ops.reduce_mean(generated_activations, 0)
    num_examples_real = math_ops.to_double(array_ops.shape(real_activations)[0])
    num_examples_generated = math_ops.to_double(
        array_ops.shape(generated_activations)[0])


    real_centered = real_activations - m
    sigma = math_ops.matmul(
        real_centered, real_centered, transpose_a=True) / (
                    num_examples_real - 1)

    gen_centered = generated_activations - m_w
    sigma_w = math_ops.matmul(
        gen_centered, gen_centered, transpose_a=True) / (
                      num_examples_generated - 1)

    sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)

    trace = math_ops.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

    mean = math_ops.reduce_sum(
        math_ops.squared_difference(m, m_w))
    fid = trace + mean
    if activations_dtype != dtypes.float64:
        fid = math_ops.cast(fid, activations_dtype)

    return fid

def kernel_classifier_distance_and_std_from_activations(real_activations,
                                                        generated_activations,
                                                        max_block_size=10,
                                                        dtype=None):

    real_activations.shape.assert_has_rank(2)
    generated_activations.shape.assert_has_rank(2)
    real_activations.shape[1].assert_is_compatible_with(
        generated_activations.shape[1])

    if dtype is None:
        dtype = real_activations.dtype
        assert generated_activations.dtype == dtype
    else:
        real_activations = math_ops.cast(real_activations, dtype)
        generated_activations = math_ops.cast(generated_activations, dtype)

    n_r = array_ops.shape(real_activations)[0]
    n_g = array_ops.shape(generated_activations)[0]

    n_bigger = math_ops.maximum(n_r, n_g)
    n_blocks = math_ops.to_int32(math_ops.ceil(n_bigger / max_block_size))

    v_r = n_r // n_blocks
    v_g = n_g // n_blocks

    n_plusone_r = n_r - v_r * n_blocks
    n_plusone_g = n_g - v_g * n_blocks

    sizes_r = array_ops.concat([
        array_ops.fill([n_blocks - n_plusone_r], v_r),
        array_ops.fill([n_plusone_r], v_r + 1),
    ], 0)
    sizes_g = array_ops.concat([
        array_ops.fill([n_blocks - n_plusone_g], v_g),
        array_ops.fill([n_plusone_g], v_g + 1),
    ], 0)

    zero = array_ops.zeros([1], dtype=dtypes.int32)
    inds_r = array_ops.concat([zero, math_ops.cumsum(sizes_r)], 0)
    inds_g = array_ops.concat([zero, math_ops.cumsum(sizes_g)], 0)

    dim = math_ops.cast(tf.shape(real_activations)[1], dtype)

    def compute_kid_block(i):
        r_s = inds_r[i]
        r_e = inds_r[i + 1]
        r = real_activations[r_s:r_e]
        m = math_ops.cast(r_e - r_s, dtype)

        g_s = inds_g[i]
        g_e = inds_g[i + 1]
        g = generated_activations[g_s:g_e]
        n = math_ops.cast(g_e - g_s, dtype)

        k_rr = (math_ops.matmul(r, r, transpose_b=True) / dim + 1)**3
        k_rg = (math_ops.matmul(r, g, transpose_b=True) / dim + 1)**3
        k_gg = (math_ops.matmul(g, g, transpose_b=True) / dim + 1)**3
        return (-2 * math_ops.reduce_mean(k_rg) +
                (math_ops.reduce_sum(k_rr) - math_ops.trace(k_rr)) / (m * (m - 1)) +
                (math_ops.reduce_sum(k_gg) - math_ops.trace(k_gg)) / (n * (n - 1)))

    ests = functional_ops.map_fn(
        compute_kid_block, math_ops.range(n_blocks), dtype=dtype, back_prop=False)

    mn = math_ops.reduce_mean(ests)

    n_blocks_ = math_ops.cast(n_blocks, dtype)
    var = control_flow_ops.cond(
        math_ops.less_equal(n_blocks, 1),
        lambda: array_ops.constant(float('nan'), dtype=dtype),
        lambda: math_ops.reduce_sum(math_ops.square(ests - mn)) / (n_blocks_ - 1))

    return mn, math_ops.sqrt(var / n_blocks_)


def inception_activations(images, num_splits=1):
    images = tf.transpose(images, [0, 2, 3, 1])
    size = 299
    images = tf.image.resize_bilinear(images, [size, size])
    generated_images_list = array_ops.split(images, num_or_size_splits=num_splits)
    activations = functional_ops.map_fn(
        fn=functools.partial(tfgan.eval.run_inception, output_tensor='pool_3:0'),
        elems=array_ops.stack(generated_images_list),
        parallel_iterations=1,
        back_prop=False,
        swap_memory=True,
        name='RunClassifier')
    activations = array_ops.concat(array_ops.unstack(activations), 0)
    return activations


def get_inception_activations(batch_size, images, inception_images, activations):
    n_batches = images.shape[0] // batch_size
    act = np.zeros([n_batches * batch_size, 2048], dtype=np.float32)
    for i in range(n_batches):
        inp = images[i * batch_size:(i + 1) * batch_size] / 255. * 2 - 1
        act[i * batch_size:(i + 1) * batch_size] = activations.eval(feed_dict={inception_images: inp})
    return act


def activations2distance(fcd, real_activation, fake_activation, act1, act2):
    return fcd.eval(feed_dict={real_activation: act1, fake_activation: act2})


def get_fid(fcd, batch_size, images1, images2, inception_images, real_activation, fake_activation, activations):
    # print('Calculating FID with %i images from each distribution' % (images1.shape[0]))
    start_time = time.time()
    act1 = get_inception_activations(batch_size, images1, inception_images, activations)
    act2 = get_inception_activations(batch_size, images2, inception_images, activations)
    fid = activations2distance(fcd, real_activation, fake_activation, act1, act2)
    # print('FID calculation time: %f s' % (time.time() - start_time))
    return fid

def get_kid(kcd, batch_size, images1, images2, inception_images, real_activation, fake_activation, activations):
    # print('Calculating KID with %i images from each distribution' % (images1.shape[0]))
    start_time = time.time()
    act1 = get_inception_activations(batch_size, images1, inception_images, activations)
    act2 = get_inception_activations(batch_size, images2, inception_images, activations)
    kcd = activations2distance(kcd, real_activation, fake_activation, act1, act2)
    # print('KID calculation time: %f s' % (time.time() - start_time))
    return kcd

def get_kid_value(kcd, act1,act2, real_activation, fake_activation):
    kcd = activations2distance(kcd, real_activation, fake_activation, act1, act2)
    return kcd

def get_images(filename):
    x = misc.imread(filename)
    x = misc.imresize(x, size=[299, 299])
    return x


