# -*- coding: utf-8 -*-
"""
Cascading model-parameter randomization sanity check (Adebayo et al.,
"Sanity Checks for Saliency Maps", NeurIPS 2018 -- the paper behind
arxiv.org/abs/1810.03292, which is what Prof. Bruno Grisci actually
linked; see experiments/paper_auc_xai_correlacao for the full
correspondence). Checks whether an attribution method's output actually
depends on the trained model's weights, or is largely insensitive to
them (behaving like an edge detector regardless of what the model
learned) -- a distinct property from the Dice/IoU "plausibility" metrics
used elsewhere in this project (Nauta et al.'s Co-12 taxonomy calls this
"correctness"; plausibility-vs-ground-truth does not test it).

Scope decision, stated plainly: the original paper randomizes literally
every layer, cascading from the output. This model's backbone
(EfficientNetV2S) is nested as ONE Keras layer (a sub-model, per
utils_gradcam.py's docstring) rather than exposed as hundreds of
top-level layers, so this implementation cascades over the model's
TOP-LEVEL components (classifier head layers, then the backbone as one
final block) rather than literally every backbone sub-layer. This still
answers the same question the check is for -- does the map degrade as
more of the model is randomized -- at a coarser but tractable
granularity, given the mini-campaign's time budget.
"""

import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr


def _randomize_top_layers(model: tf.keras.Model, n_from_output: int) -> tf.keras.Model:
    """
    Return a NEW model, same architecture, where the last `n_from_output`
    entries of `model.layers` (counting from the output side) have fresh,
    randomly-initialized weights (via tf.keras.models.clone_model, which
    reinitializes each layer using its own configured initializer -- so a
    Conv2D gets Conv2D-appropriate random weights, a BatchNorm gets
    identity-like init, etc., not one generic noise distribution for
    everything), and every other layer keeps `model`'s current (trained)
    weights, copied over explicitly.
    """
    fresh = tf.keras.models.clone_model(model)
    layers_orig = model.layers
    layers_fresh = fresh.layers
    n = len(layers_orig)
    randomize_from_index = n - n_from_output  # layers at this index or later stay randomized
    for i, (lo, lf) in enumerate(zip(layers_orig, layers_fresh)):
        if i >= randomize_from_index:
            continue  # leave fresh (random) weights as-is
        w = lo.get_weights()
        if w:
            lf.set_weights(w)
    return fresh


def cascading_randomization_test(model: tf.keras.Model, images: np.ndarray,
                                  xai_method_func, target_size=(299, 299),
                                  n_steps: int = None, **method_kwargs) -> dict:
    """
    Adebayo et al.'s sanity check: compute the attribution map with the
    fully-trained model (step 0, the reference), then progressively
    randomize more of the model's top-level components (from the output
    side inward) and recompute the map at each step, measuring how much
    it still agrees with the reference (mean Spearman rank correlation
    across `images`, flattening each H*W map to a vector).

    A map that STAYS highly correlated even at full randomization is
    evidence the method isn't actually reflecting what the model learned
    (fails the check, per the paper); a map whose correlation drops
    toward 0 as more of the model is destroyed is evidence it does
    (passes).

    Args:
        model: Trained Keras model to test
        images: Small batch of images to average the check over (this is
            diagnostic, not a metric to report per-image -- a handful of
            images, e.g. 10-20, is enough)
        xai_method_func: One of generate_gradcam / generate_lime /
            generate_occlusion_sensitivity from utils_gradcam.py /
            utils_xai_extra.py -- called as
            xai_method_func(model, images, target_size=target_size, **method_kwargs)
        target_size: Passed through to xai_method_func
        n_steps: How many randomization steps between 0 (none) and
            len(model.layers) (everything) to test, evenly spaced. None
            (default) tests every top-level layer boundary individually
            (len(model.layers)+1 points, i.e. as fine-grained as this
            top-level-only scope allows).
        **method_kwargs: Forwarded to xai_method_func (e.g. num_samples
            for LIME, patch_size/stride for Occlusion)

    Returns:
        {n_layers_randomized: mean_spearman_rho}, ordered from least to
        most randomized. n_layers_randomized=0 is trivially rho=1.0
        (compared against itself) -- included as a sanity check on the
        correlation computation itself, not a finding.
    """
    total_layers = len(model.layers)
    if n_steps is None:
        steps = list(range(0, total_layers + 1))
    else:
        steps = sorted(set(int(round(x)) for x in np.linspace(0, total_layers, n_steps)))

    reference_maps = xai_method_func(model, images, target_size=target_size, **method_kwargs)

    results = {}
    for n_random in steps:
        if n_random == 0:
            maps = reference_maps
        else:
            randomized_model = _randomize_top_layers(model, n_random)
            maps = xai_method_func(randomized_model, images, target_size=target_size, **method_kwargs)

        rhos = []
        for i in range(len(images)):
            rho, _ = spearmanr(reference_maps[i].ravel(), maps[i].ravel())
            if not np.isnan(rho):
                rhos.append(rho)
        results[n_random] = float(np.mean(rhos)) if rhos else float("nan")

    return results
