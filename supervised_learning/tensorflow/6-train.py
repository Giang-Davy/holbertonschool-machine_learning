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
    """
    Builds, trains, and saves a neural network classifier.

    Args:
        X_train: numpy.ndarray - Training input data.
        Y_train: numpy.ndarray - Training labels.
        X_valid: numpy.ndarray - Validation input data.
        Y_valid: numpy.ndarray - Validation labels.
        layer_sizes: list - Number of nodes in each layer of the network.
        activations: list - Activation functions for each layer.
        alpha: float - Learning rate.
        iterations: int - Number of iterations for training.
        save_path: str - Path to save the trained model.

    Returns:
        str: The path where the model was saved.
    """
    # Create placeholders
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])

    # Build forward propagation
    y_pred = forward_prop(x, layer_sizes, activations)

    # Calculate loss and accuracy
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)

    # Create training operation
    train_op = create_train_op(loss, alpha)

    # Initialize variables
    init = tf.global_variables_initializer()

    # Save model
    saver = tf.train.Saver()

    # Start session
    with tf.Session() as sess:
        sess.run(init)

        for i in range(iterations + 1):
            # Perform training step
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})

            if i % 100 == 0 or i == iterations:
                # Evaluate metrics
                cost_train, accuracy_train = sess.run(
                    [loss, accuracy],
                    feed_dict={x: X_train, y: Y_train})
                cost_valid, accuracy_valid = sess.run(
                    [loss, accuracy],
                    feed_dict={x: X_valid, y: Y_valid})

                # Print metrics
                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {cost_train}")
                print(f"\tTraining Accuracy: {accuracy_train}")
                print(f"\tValidation Cost: {cost_valid}")
                print(f"\tValidation Accuracy: {accuracy_valid}")

        # Save the model
        saved_path = saver.save(sess, save_path)

    return saved_path
