# -*- coding: utf-8 -*-
"""
Grad-CAM for the EfficientNetV2S-based binary classifier built in
dr_hcpa_v2_2024.py (build_model): Input -> EfficientNetV2S(include_top=False,
submodel) -> GAP -> Dropout -> Dense(512, elu) -> Dropout -> Dense(1, sigmoid).

Backbone-agnostic by design (finds the nested backbone submodel and its last
4D-output layer generically), so this file needed no logic changes when the
backbone was swapped from InceptionV3 to EfficientNetV2S — only these
docstrings were updated.

Used by script3_avalia.py to produce saliency maps compared against the real
FGADR lesion masks (data/Seg-set/*_Masks) for the XAI metrics (Dice, IoU,
Pointing Game Accuracy).
"""

import numpy as np
import tensorflow as tf


def _get_backbone_submodel(model: tf.keras.Model) -> tf.keras.Model:
    """Find the nested backbone functional submodel (e.g. EfficientNetV2S) inside `model`."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise ValueError("No nested backbone submodel (e.g. EfficientNetV2S) found in model")


def find_last_conv_layer(submodel: tf.keras.Model) -> str:
    """
    Return the name of the last layer in `submodel` whose output is a 4D
    tensor (batch, H, W, C) — i.e. the last convolutional-like feature map,
    used as the Grad-CAM target. Not hardcoded to a specific layer name
    (e.g. 'mixed10') so it stays robust across Keras/TF versions.
    """
    for layer in reversed(submodel.layers):
        shape = getattr(layer, "output_shape", None)
        if shape is None:
            continue
        if isinstance(shape, list):
            shape = shape[0]
        if shape is not None and len(shape) == 4:
            return layer.name
    raise ValueError("No 4D (convolutional) layer found in backbone submodel")


def generate_gradcam(model: tf.keras.Model, images: np.ndarray, target_size=(299, 299)) -> np.ndarray:
    """
    Compute Grad-CAM heatmaps for a batch of images against a binary
    sigmoid classifier.

    Args:
        model: Trained Keras model (Input -> backbone -> GAP -> Dense(1, sigmoid))
        images: Batch of preprocessed images, shape (N, H, W, 3), float32
        target_size: Output heatmap resolution (W, H) to resize to

    Returns:
        Heatmaps normalized to [0, 1], shape (N, target_size[1], target_size[0])
    """
    import cv2

    backbone = _get_backbone_submodel(model)
    last_conv_name = find_last_conv_layer(backbone)

    # Split the forward pass in two: (1) frozen backbone up to the last conv
    # feature map, (2) the trainable classifier head (GAP + Dense) applied on
    # top of it. This is necessary because the backbone's conv weights are
    # frozen (trainable=False), so GradientTape never auto-watches them —
    # explicitly watching the intermediate conv feature tensor itself (before
    # it feeds the head) is what lets gradients flow back to it.
    #
    # conv_layer_model is built from the backbone's OWN input/output pair
    # (not model.inputs) — bridging from the outer model's input to a layer
    # inside the nested submodel works for a freshly-built model, but breaks
    # ("Graph disconnected") once the model has been saved and reloaded, since
    # Keras reconstructs the nested submodel with its own separate Input node.
    conv_layer_model = tf.keras.models.Model(
        inputs=backbone.input, outputs=backbone.get_layer(last_conv_name).output
    )
    backbone_index = model.layers.index(backbone)
    head_input = tf.keras.Input(shape=conv_layer_model.output.shape[1:])
    head_output = head_input
    for layer in model.layers[backbone_index + 1:]:
        head_output = layer(head_output)
    head_model = tf.keras.models.Model(inputs=head_input, outputs=head_output)

    images_tensor = tf.convert_to_tensor(images, dtype=tf.float32)
    conv_output = conv_layer_model(images_tensor, training=False)

    with tf.GradientTape() as tape:
        tape.watch(conv_output)
        predictions = head_model(conv_output, training=False)
        # Binary sigmoid: the class score IS the predicted probability.
        class_scores = predictions[:, 0]

    grads = tape.gradient(class_scores, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))  # (N, C)
    pooled_grads = pooled_grads[:, tf.newaxis, tf.newaxis, :]

    heatmaps = tf.reduce_sum(conv_output * pooled_grads, axis=-1)  # (N, h, w)
    heatmaps = tf.nn.relu(heatmaps).numpy()

    resized = np.zeros((heatmaps.shape[0], target_size[1], target_size[0]), dtype=np.float32)
    for i in range(heatmaps.shape[0]):
        heatmap = heatmaps[i]
        max_val = heatmap.max()
        if max_val > 0:
            heatmap = heatmap / max_val
        resized[i] = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)

    return resized
