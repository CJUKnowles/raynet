"""TensorFlow v1 compatibility setup required by the original Astraea learner."""

import os
import sys
import types


if os.getenv("ASTRAEA_TF_USE_GPU", "0") != "1":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/astraea-matplotlib")

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def _unique_layer_name(base_name):
    graph = tf.get_default_graph()
    counters = getattr(graph, "_astraea_layer_name_counters", None)
    if counters is None:
        counters = {}
        graph._astraea_layer_name_counters = counters

    key = (tf.get_variable_scope().name, base_name)
    index = counters.get(key, 0)
    counters[key] = index + 1
    return base_name if index == 0 else f"{base_name}_{index}"


def _dense(inputs, units, activation=None, name=None):
    input_dim = inputs.shape.as_list()[-1]
    if input_dim is None:
        raise ValueError("Dense input dimension must be statically known")

    # The original critic is built twice under AUTO_REUSE and expects its unnamed output layer to be shared by both calls.
    with tf.variable_scope(name or "dense", reuse=tf.AUTO_REUSE):
        kernel = tf.get_variable(
            "kernel",
            shape=[input_dim, units],
            initializer=tf.glorot_uniform_initializer(),
        )
        bias = tf.get_variable("bias", shape=[units], initializer=tf.zeros_initializer())
        output = tf.matmul(inputs, kernel) + bias
        return activation(output) if activation is not None else output


def _batch_normalization(inputs, training=False, scale=True, name=None):
    dim = inputs.shape.as_list()[-1]
    if dim is None:
        raise ValueError("Batch normalization input dimension must be statically known")

    layer_name = name or _unique_layer_name("batch_normalization")
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        beta = tf.get_variable("beta", shape=[dim], initializer=tf.zeros_initializer())
        gamma = None
        if scale:
            gamma = tf.get_variable("gamma", shape=[dim], initializer=tf.ones_initializer())
        moving_mean = tf.get_variable(
            "moving_mean",
            shape=[dim],
            initializer=tf.zeros_initializer(),
            trainable=False,
        )
        moving_variance = tf.get_variable(
            "moving_variance",
            shape=[dim],
            initializer=tf.ones_initializer(),
            trainable=False,
        )

        mean, variance = tf.nn.moments(inputs, axes=[0])
        tf.add_to_collection(
            tf.GraphKeys.UPDATE_OPS,
            tf.assign(moving_mean, moving_mean * 0.99 + mean * 0.01),
        )
        tf.add_to_collection(
            tf.GraphKeys.UPDATE_OPS,
            tf.assign(moving_variance, moving_variance * 0.99 + variance * 0.01),
        )

        train_output = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, 1e-3)
        infer_output = tf.nn.batch_normalization(inputs, moving_mean, moving_variance, beta, gamma, 1e-3)
        if isinstance(training, bool):
            return train_output if training else infer_output
        return tf.cond(training, lambda: train_output, lambda: infer_output)


tf.layers = types.SimpleNamespace(
    dense=_dense,
    batch_normalization=_batch_normalization,
)
sys.modules["tensorflow"] = tf
