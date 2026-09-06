# -*- coding: utf-8 -*-
"""
LIME and Occlusion Sensitivity for the same binary EfficientNetV2S
classifier `utils_gradcam.py` targets (Input -> backbone -> GAP ->
Dropout -> Dense(512, elu) -> Dropout -> Dense(1, sigmoid)).

Added to compare Grad-CAM against two mechanistically different
attribution families (per Molnar's taxonomy and the XAI methodology
review's correspondence with Prof. Bruno Grisci -- see
experiments/paper_auc_xai_correlacao): Grad-CAM is gradient-based;
Occlusion Sensitivity is perturbation-based (direct sensitivity to
masking image regions); LIME is surrogate-model-based (fits a local
interpretable linear model on perturbed samples) -- three genuinely
different mechanisms, not variations on one. An earlier version of this
file used Integrated Gradients instead of LIME, but IG is also
gradient-based and was swapped out on Prof. Grisci's explicit
recommendation for more method diversity (SHAP was considered and
rejected: a model-agnostic SHAP variant for images, e.g. KernelSHAP,
costs even more forward passes per image than Occlusion already does,
which was already the campaign's compute bottleneck).

Both functions share `generate_gradcam`'s exact signature and output
convention -- (N, H, W) float32, normalized to [0, 1] per image -- so
they plug directly into utils_metrics.py's dice_coefficient/iou_score/
pointing_game_accuracy with no changes needed there.
"""

import numpy as np
import tensorflow as tf


def generate_lime(model: tf.keras.Model, images: np.ndarray,
                   target_size=(299, 299), num_samples: int = 150,
                   num_segments: int = 50, batch_size: int = 64) -> np.ndarray:
    """
    LIME (Ribeiro et al., 2016): segments the image into superpixels,
    generates `num_samples` random on/off perturbations of those
    superpixels, fits a local linear surrogate model on how the
    classifier's predicted probability responds to each perturbation, and
    uses the surrogate's per-superpixel coefficients as the importance
    map -- a fundamentally different mechanism from both gradient-based
    (Grad-CAM) and direct-perturbation-sensitivity (Occlusion) methods.

    Args:
        model: Trained Keras model (Input -> backbone -> ... -> Dense(1, sigmoid))
        images: Batch of preprocessed images, shape (N, H, W, 3), float32
            in [0, 255] (matches this codebase's convention -- EfficientNetV2S
            bundles its own rescaling as the first model layer)
        target_size: Output heatmap resolution (W, H)
        num_samples: Perturbed samples LIME draws per image to fit the local
            surrogate model. The library's own default (1000) was judged too
            expensive across 10 filters x 10 repetitions x ~100 images each
            (same cost concern as Occlusion's patch grid -- see that
            function's docstring); 150 is a commonly used lighter-weight
            setting that still gives a stable fit given `num_segments` below.
        num_segments: Target superpixel count (SLIC segmentation, fixed
            count rather than quickshift's data-dependent count, for
            predictable/bounded runtime).
        batch_size: Perturbed images per forward pass while LIME queries
            the model internally.

    Returns:
        Heatmaps normalized to [0, 1], shape (N, target_size[1], target_size[0])
    """
    import cv2
    from lime import lime_image
    from lime.wrappers.scikit_image import SegmentationAlgorithm

    segmenter = SegmentationAlgorithm("slic", n_segments=num_segments,
                                       compactness=10, sigma=1, start_label=0)
    explainer = lime_image.LimeImageExplainer()

    n = images.shape[0]
    h, w = images.shape[1], images.shape[2]
    heatmaps = np.zeros((n, h, w), dtype=np.float32)

    def classifier_fn(batch_images: np.ndarray) -> np.ndarray:
        # LIME expects a (samples, num_classes) probability matrix, like a
        # softmax classifier -- wrap this model's single sigmoid output
        # [p(referable)] as [[1-p, p], ...] so column 1 is always the same
        # "referable" class score every other method here explains too
        # (not "whichever class LIME happens to pick"; see the fixed
        # `labels=(1,)` below).
        preds = model(tf.convert_to_tensor(batch_images, dtype=tf.float32),
                       training=False).numpy()[:, 0]
        return np.stack([1.0 - preds, preds], axis=1)

    for i in range(n):
        image = images[i].astype(np.float64)  # skimage segmentation expects float/uint
        try:
            explanation = explainer.explain_instance(
                image, classifier_fn, labels=(1,), hide_color=0,
                num_samples=num_samples, segmentation_fn=segmenter,
                batch_size=batch_size,
            )
        except Exception:
            # A pathological image (e.g. segmentation collapses to a single
            # superpixel) shouldn't crash the whole evaluation run -- leave
            # this image's heatmap as all-zero (same fallback shape/dtype
            # contract as every other method here) and move on.
            continue

        segments = explanation.segments
        heat = np.zeros((h, w), dtype=np.float32)
        # local_exp[1]: list of (segment_id, weight) for the "referable"
        # class explained above. Negative weights (evidence AGAINST
        # referable) are clipped to 0, matching Grad-CAM's ReLU and
        # Occlusion's max(drop, 0) -- only keep evidence FOR the class.
        for segment_id, weight in explanation.local_exp[1]:
            if weight > 0:
                heat[segments == segment_id] = weight

        heatmaps[i] = heat

    resized = np.zeros((n, target_size[1], target_size[0]), dtype=np.float32)
    for i in range(n):
        heatmap = heatmaps[i]
        max_val = heatmap.max()
        if max_val > 0:
            heatmap = heatmap / max_val
        resized[i] = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)

    return resized


def generate_occlusion_sensitivity(model: tf.keras.Model, images: np.ndarray,
                                    target_size=(299, 299), patch_size: int = 32,
                                    stride: int = 16, occlusion_value: float = 0.0,
                                    batch_size: int = 64) -> np.ndarray:
    """
    Occlusion Sensitivity (Zeiler & Fergus, 2014): slide a square occluding
    patch over the image on a `stride`-spaced grid; at each position, the
    drop in predicted probability (relative to the unoccluded prediction)
    is the importance score assigned to that patch. Perturbation-based --
    does not use gradients, so it probes the model through a completely
    different mechanism than Grad-CAM or Integrated Gradients.

    Deliberately more expensive than the other two methods (each grid
    position costs one extra forward pass): patch_size=32/stride=16 on a
    299x299 image gives ~18x18=324 positions/image. Called only on a
    fixed image subsample (not the full 553-image test set) in the XAI
    methodology review's mini-campaign, for exactly this reason.

    Args:
        model: Trained Keras model
        images: Batch of preprocessed images, shape (N, H, W, 3), float32
        target_size: Output heatmap resolution (W, H)
        patch_size: Side length (px) of the square occlusion patch
        stride: Step (px) between adjacent patch positions (< patch_size
            gives overlapping patches, i.e. finer/smoother heatmaps at
            proportionally higher cost)
        occlusion_value: Pixel value the patch is filled with (0.0 = black,
            matching Integrated Gradients' baseline for consistency)
        batch_size: How many occluded copies to forward-pass at once

    Returns:
        Heatmaps normalized to [0, 1], shape (N, target_size[1], target_size[0])
    """
    import cv2

    images_tensor = tf.convert_to_tensor(images, dtype=tf.float32)
    n, h, w, c = images_tensor.shape
    h, w = int(h), int(w)

    ys = list(range(0, h - patch_size + 1, stride)) or [0]
    xs = list(range(0, w - patch_size + 1, stride)) or [0]
    if ys[-1] + patch_size < h:
        ys.append(h - patch_size)
    if xs[-1] + patch_size < w:
        xs.append(w - patch_size)

    base_preds = model(images_tensor, training=False).numpy()[:, 0]  # (N,)

    heatmaps = np.zeros((n, h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    for i in range(n):
        image = images_tensor[i]
        occluded_batch, positions = [], []
        for y in ys:
            for x in xs:
                mask = np.ones((h, w, 1), dtype=np.float32)
                mask[y:y + patch_size, x:x + patch_size, :] = 0.0
                occluded = image.numpy() * mask + occlusion_value * (1.0 - mask)
                occluded_batch.append(occluded)
                positions.append((y, x))

        drops = np.zeros(len(positions), dtype=np.float32)
        for start in range(0, len(occluded_batch), batch_size):
            chunk = np.stack(occluded_batch[start:start + batch_size], axis=0)
            preds = model(tf.convert_to_tensor(chunk, dtype=tf.float32), training=False).numpy()[:, 0]
            drops[start:start + len(preds)] = base_preds[i] - preds

        heat = np.zeros((h, w), dtype=np.float32)
        cnt = np.zeros((h, w), dtype=np.float32)
        for (y, x), drop in zip(positions, drops):
            # Negative drops (occlusion INCREASED the score) are clipped to
            # 0 -- Occlusion Sensitivity, like Grad-CAM's ReLU, only keeps
            # evidence FOR the predicted class, not against it.
            heat[y:y + patch_size, x:x + patch_size] += max(drop, 0.0)
            cnt[y:y + patch_size, x:x + patch_size] += 1.0
        cnt[cnt == 0] = 1.0
        heatmaps[i] = heat / cnt

    resized = np.zeros((n, target_size[1], target_size[0]), dtype=np.float32)
    for i in range(n):
        heatmap = heatmaps[i]
        max_val = heatmap.max()
        if max_val > 0:
            heatmap = heatmap / max_val
        resized[i] = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)

    return resized
