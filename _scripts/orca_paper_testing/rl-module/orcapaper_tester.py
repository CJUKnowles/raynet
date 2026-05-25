import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from agent import Agent

CHECKPOINT = "/home/james/raynet/_models/Orca-paper/model.ckpt-1283529"

tf.reset_default_graph()

sess = tf.Session()

agent = Agent(
    s_dim=70,
    a_dim=1,
    h1_shape=256,
    h2_shape=256,
)

agent.build_learn()

agent.saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

agent.assign_sess(sess)

agent.saver.restore(sess, CHECKPOINT)

print("checkpoint loaded")