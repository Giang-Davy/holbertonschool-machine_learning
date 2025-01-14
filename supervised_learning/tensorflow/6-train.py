#!/usr/bin/env python3
"""fonction"""


import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    # Create placeholders
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])

    # Forward propagation
    y_pred = forward_prop(x, layer_sizes, activations)

    # Compute cost and accuracy
    cost = calculate_loss(y_pred, y)
    accuracy = calculate_accuracy(y_pred, y)

    # Create train operation
    train_op = create_train_op(cost, alpha)

    # Initialize variables
    init = tf.global_variables_initializer()

    # Start a session
    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)

        # Training loop
        for i in range(iterations + 1):
            # Train the model
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})

            if i % 100 == 0 or i == 0 or i == iterations:
                # Calculate costs and accuracy for training and validation sets
                train_cost, train_accuracy = sess.run([cost, accuracy], feed_dict={x: X_train, y: Y_train})
                valid_cost, valid_accuracy = sess.run([cost, accuracy], feed_dict={x: X_valid, y: Y_valid})

                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {train_cost}")
                print(f"\tTraining Accuracy: {train_accuracy}")
                print(f"\tValidation Cost: {valid_cost}")
                print(f"\tValidation Accuracy: {valid_accuracy}")

        # Save the model
        saver = tf.train.Saver()
        saver.save(sess, save_path)

    return save_path
