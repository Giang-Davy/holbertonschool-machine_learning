#!/usr/bin/env python3
import tensorflow as tf

def l2_reg_cost(cost, model):
	l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in model.trainable_variables if 'kernel' in var.name])
	return cost + tf.broadcast_to(l2_loss, cost.shape)
