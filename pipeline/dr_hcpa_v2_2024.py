# -*- coding: utf-8 -*-
"""
Treinamento binário (referável DR) sobre EfficientNetV2S, em duas fases
(warm-up do topo + fine-tuning global), replicando a metodologia de
Kao & Lin (2024) adaptada para classificação binária e input 299x299
(tamanho nativo deste pipeline — o artigo original usa 256x256).
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
"""
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import re
from os import makedirs
import time
import argparse
from tensorflow.keras import applications
print(tf.__version__)

# Infra fix (not a scientific/methodological change): grappler's
# layout_optimizer pass (NHWC<->NCHW graph rewrite for GPU efficiency) hits a
# known TensorFlow bug against EfficientNetV2's dropout/SelectV2 op on this
# GPU/driver/TF-2.13 combination — confirmed via real repeated runs during
# infrastructure validation: "layout failed: INVALID_ARGUMENT: Size of
# values 0 does not match size of permutation 4 @ fanin shape in
# model/efficientnetv2-s/block1b_drop/dropout/SelectV2-2-..." immediately
# followed by a hard C++ abort (std::bad_function_call, exit code -6/SIGABRT)
# that kills the whole training process with no Python traceback. Disabling
# only this one grappler pass (not XLA, not the other grappler
# optimizations) avoids the crash; it does not change computed values,
# architecture, or hyperparameters.
tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

# Hiperparâmetros mandatados pela metodologia replicada (Kao & Lin, 2024) /
# pela auditoria + BUSCA de hiperparâmetros executada em 2026-07-09 (ver
# experiments/MODEL_OPTIMIZATION_AUDIT.md e experiments/hpo_best_config.json
# para os 24 trials reais e a decisão de confirmação: Δ=+0.0157 AUC,
# ~3.3x o desvio-padrão combinado entre seeds — vencedor real, não estimado).
# PHASE2_LR e LABEL_SMOOTHING abaixo já refletem a configuração vencedora
# adotada; os defaults de --optimizer/--weight_decay/--clipnorm/
# --finetune_depth/--use_class_weights/--batch_size (mais abaixo, no
# argparse) também foram atualizados. Todos os itens seguem expostos via
# CLI (pipeline/hpo_search.py) para permitir buscas futuras, mas a matriz
# oficial de 576 filtros SEMPRE usa os defaults (nunca passa essas flags
# explicitamente), de modo que todas as filtro x repetição continuem
# estritamente comparáveis entre si.
PHASE1_EPOCHS = 5
PHASE1_LR = 0.001
PHASE2_EPOCHS = 50
PHASE2_LR = 0.0001  # HPO 2026-07-09: vencedor (era 0.00002)
LABEL_SMOOTHING = 0.1  # HPO 2026-07-09: vencedor (era 0.2, alto demais vs. literatura)
REDUCE_LR_FACTOR = 0.1
REDUCE_LR_PATIENCE = 5
EARLY_STOPPING_PATIENCE = 10


def parse_args():
    parser = argparse.ArgumentParser(description='This script is used for training the hcpa model using dataset from tfrecord files. See more: python3 dr_hcpa_v2_2024.py -h')
    parser.add_argument('--tfrec_dir', type=str, default='./data/all', help='Directory containing TFRecord files')
    parser.add_argument('--dataset', type=str, default='all', help='Name of the dataset')
    parser.add_argument('--results', type=str, default='./results/all', help='Directory to save results')
    parser.add_argument('--exec', type=int, default=0, help='Execution number')
    parser.add_argument('--img_sizes', type=int, default=299, help='Image sizes (native size of this pipeline; not the 256 used in the reference paper)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (HPO 2026-07-09 winner: 32, was 16 pre-audit — see '
                             'experiments/MODEL_OPTIMIZATION_AUDIT.md).')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes (must be 2 — this pipeline only supports binary referable-DR classification)')
    parser.add_argument('--verbose', type=int, default=1, help='Verbose level for training')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (controls weight init and data shuffling — should be set to the repetition index)')
    parser.add_argument('--use_class_weights', action=argparse.BooleanOptionalAction, default=True,
                        help='Weight the loss inversely proportional to class frequency, computed from '
                             'the TRAINING split only (never test). Default ON as of the HPO run of '
                             '2026-07-09 (was opt-in/OFF pre-audit) — FGADR referable-DR binarization is '
                             '~83%%/17%% imbalanced; see experiments/MODEL_OPTIMIZATION_AUDIT.md. Pass '
                             '--no-use_class_weights to restore the pre-audit behavior.')
    parser.add_argument('--phase1_epochs', type=int, default=None,
                        help='TEST-ONLY override for the phase-1 (warm-up) epoch count. Defaults to the '
                             'paper-mandated PHASE1_EPOCHS (5) when omitted — must NEVER be passed for the '
                             'official experimental matrix, only for infrastructure/integration smoke tests.')
    parser.add_argument('--phase2_epochs', type=int, default=None,
                        help='TEST-ONLY override for the phase-2 (fine-tuning) epoch count. Defaults to the '
                             'paper-mandated PHASE2_EPOCHS (50) when omitted — must NEVER be passed for the '
                             'official experimental matrix, only for infrastructure/integration smoke tests.')
    parser.add_argument('--phase1_lr', type=float, default=None,
                        help='HPO override for the phase-1 (warm-up) learning rate. Default None -> '
                             'uses the paper-mandated PHASE1_LR (0.001).')
    parser.add_argument('--phase2_lr', type=float, default=None,
                        help='HPO override for the phase-2 (fine-tuning) BASE learning rate (fed to '
                             '--optimizer and, when --phase2_scheduler=cosine_warmup, to the schedule). '
                             'Default None -> uses PHASE2_LR (0.0001, HPO 2026-07-09 winner).')
    parser.add_argument('--early_stopping_patience', type=int, default=None,
                        help='HPO override for phase-2 EarlyStopping patience. Default None -> uses '
                             'the paper-mandated EARLY_STOPPING_PATIENCE (10). hpo_search.py uses a '
                             'tighter patience for screening trials to control cost.')

    # --- Hyperparameter-optimization flags (see experiments/MODEL_OPTIMIZATION_AUDIT.md) ---
    # Every default below reproduces the EXACT pre-audit behavior; the 576-filter
    # campaign must never pass any of these explicitly, only hpo_search.py does.
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer family (HPO 2026-07-09 winner: adamw, was adam pre-audit). '
                             'RMSProp intentionally not offered — see audit doc for why.')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Decoupled weight decay for --optimizer=adamw (HPO 2026-07-09 winner: '
                             '0.0001, was 0.0/unused pre-audit).')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum (only used when --optimizer=sgd — sgd was NOT the HPO winner, '
                             'see audit doc; kept available for future ablation).')
    parser.add_argument('--clipnorm', type=float, default=1.0,
                        help='Gradient clipping (global norm). Adopted directly (not a search axis, '
                             'see audit doc) as a stability safeguard; was None (no clipping) pre-audit.')
    parser.add_argument('--use_ema', action=argparse.BooleanOptionalAction, default=False,
                        help='Exponential moving average of weights (Keras optimizer use_ema). '
                             'Stage-2 HPO search axis (see experiments/MODEL_OPTIMIZATION_AUDIT.md). '
                             'Default OFF (the Stage-1 winning config).')
    parser.add_argument('--ema_momentum', type=float, default=0.99,
                        help='EMA decay (only used when --use_ema is set).')
    parser.add_argument('--phase2_scheduler', type=str, default='reduce_on_plateau',
                        choices=['reduce_on_plateau', 'cosine_warmup'],
                        help='Phase-2 LR schedule (default: reduce_on_plateau, the pre-audit behavior). '
                             'cosine_warmup is time-based (does not read val_loss — see audit doc for '
                             'why that matters given the known test-as-validation caveat).')
    parser.add_argument('--warmup_steps_frac', type=float, default=0.05,
                        help='Fraction of total phase-2 steps used for linear LR warmup '
                             '(only used when --phase2_scheduler=cosine_warmup).')
    parser.add_argument('--finetune_depth', type=str, default='partial', choices=['full', 'partial'],
                        help="HPO 2026-07-09 winner: 'partial' (was 'full' pre-audit). 'full': unfreeze "
                             "the entire backbone in phase 2. 'partial': keep the first "
                             "PARTIAL_FREEZE_FRACTION of backbone layers (by index, earliest/most generic "
                             "features) frozen, unfreeze only the rest.")
    parser.add_argument('--partial_freeze_fraction', type=float, default=0.7,
                        help='Fraction of backbone layers (from the input side) kept frozen when '
                             '--finetune_depth=partial.')
    parser.add_argument('--label_smoothing', type=float, default=None,
                        help='Phase-2 label smoothing. Default None -> uses the LABEL_SMOOTHING '
                             'constant (0.1, HPO 2026-07-09 winner — was 0.2 pre-audit).')
    parser.add_argument('--augment', action='store_true', default=False,
                        help='Enable GEOMETRIC-ONLY augmentation (random flip/rotation/translation), '
                             'applied identically regardless of preprocessing filter. Default OFF: the '
                             'HPO run of 2026-07-09 found augmentation ON scored LOWER on the internal '
                             'validation (0.901 vs 0.929 AUC in screening) — an honest empirical result, '
                             'not assumed from literature. Deliberately excludes any color/brightness/'
                             'contrast augmentation regardless, which would confound the 576-filter '
                             'photometric-preprocessing comparison — see audit doc.')
    parser.add_argument('--mixed_precision', type=str, default='off', choices=['off', 'mixed_float16', 'mixed_bfloat16'],
                        help="Global Keras mixed-precision policy. Default 'off' (the pre-audit behavior, "
                             "float32 throughout).")

    return parser.parse_args()

args = parse_args()
# Accessing arguments
TFREC_DIR = args.tfrec_dir
dataset = args.dataset
results = args.results
exec = args.exec
IMG_SIZES = args.img_sizes
BATCH_SIZE = args.batch_size
NUM_CLASSES = args.num_classes
VERBOSE = args.verbose
USE_CLASS_WEIGHTS = args.use_class_weights

# TEST-ONLY overrides (see argparse help above) — default (None) preserves
# the paper-mandated PHASE1_EPOCHS/PHASE2_EPOCHS constants untouched, so the
# official matrix (which never passes these flags) is unaffected.
if args.phase1_epochs is not None:
    PHASE1_EPOCHS = args.phase1_epochs
if args.phase2_epochs is not None:
    PHASE2_EPOCHS = args.phase2_epochs
if args.phase1_lr is not None:
    PHASE1_LR = args.phase1_lr
if args.phase2_lr is not None:
    PHASE2_LR = args.phase2_lr
if args.early_stopping_patience is not None:
    EARLY_STOPPING_PATIENCE = args.early_stopping_patience
if args.label_smoothing is not None:
    LABEL_SMOOTHING = args.label_smoothing

# Comprehensive parameter dump — every Stage-2-relevant hyperparameter
# printed explicitly to the job log, so the log ITSELF is a complete,
# self-contained record of what actually ran (never requires cross-
# referencing the launcher's own config dict to know, e.g., whether
# gradient clipping or EMA were active in a given run — see Stage-2 plan
# §9 pre-submission audit: "os logs registram corretamente todos os
# parâmetros").
print('#### ==== HYPERPARAMETERS IN EFFECT ====')
print(f'#### optimizer={args.optimizer} phase1_lr={PHASE1_LR} phase2_lr={PHASE2_LR} '
      f'weight_decay={args.weight_decay} momentum={args.momentum} clipnorm={args.clipnorm}')
print(f'#### phase2_scheduler={args.phase2_scheduler} warmup_steps_frac={args.warmup_steps_frac}')
print(f'#### finetune_depth={args.finetune_depth} partial_freeze_fraction={args.partial_freeze_fraction}')
print(f'#### label_smoothing={LABEL_SMOOTHING} use_class_weights={args.use_class_weights} '
      f'batch_size={args.batch_size}')
print(f'#### augment={args.augment} mixed_precision={args.mixed_precision} '
      f'use_ema={args.use_ema} ema_momentum={args.ema_momentum}')
print(f'#### phase1_epochs={PHASE1_EPOCHS} phase2_epochs={PHASE2_EPOCHS} '
      f'early_stopping_patience={EARLY_STOPPING_PATIENCE} seed={args.seed}')
try:
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils_common import get_git_commit_hash
    import logging as _logging
    _git_commit = get_git_commit_hash(_logging.getLogger(__name__))
except Exception:
    _git_commit = ""
import sklearn as _sklearn
import scipy as _scipy
print(f'#### git_commit={_git_commit} tensorflow={tf.__version__} sklearn={_sklearn.__version__} '
      f'scipy={_scipy.__version__} numpy={np.__version__}')
print('#### =======================================')

# Global mixed-precision policy — must be set before build_model() creates
# any layers/variables. Default 'off' reproduces the pre-audit float32
# behavior exactly (see experiments/MODEL_OPTIMIZATION_AUDIT.md).
if args.mixed_precision != 'off':
    tf.keras.mixed_precision.set_global_policy(args.mixed_precision)
    print(f'#### Mixed precision policy: {args.mixed_precision}')

# The model below is always a single-neuron sigmoid classifier (binary
# referable-DR). --num_classes exists for interface compatibility but was
# previously silently ignored if passed as anything else — fail loudly instead.
assert NUM_CLASSES == 2, (
    f"--num_classes={NUM_CLASSES} is not supported: this pipeline is binary-only "
    f"(referable DR): ICDR grades 0,1 -> 0; 2,3,4 -> 1."
)

# Seed the RNGs actually used for weight init and dataset shuffling.
import random
random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

IMAGE_SIZE = [IMG_SIZES, IMG_SIZES]
kepsilon = 1e-7
SHOW_FILES = False

'''
Create folder for output.
'''
makedirs(results, exist_ok=True)

def detect_hardware():
  try:
    tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
    print('Running on TPU ', tpu_resolver.master())
  except ValueError:
    tpu_resolver = None
    gpus = tf.config.experimental.list_logical_devices("GPU")

  # Select appropriate distribution strategy
  if tpu_resolver:
    tf.config.experimental_connect_to_cluster(tpu_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
    strategy = tf.distribute.TPUStrategy(tpu_resolver)
#     print('Running on TPU ', tpu_resolver.cluster_spec().as_dict()['worker'])
  elif len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
    print('Running on multiple GPUs ', [gpu.name for gpu in gpus])
  elif len(gpus) == 1:
    strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
    print('Running on single GPU ', gpus[0].name)
  else:
    strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
    print('Running on CPU')

  return strategy


strategy = detect_hardware()
REPLICAS = strategy.num_replicas_in_sync

print(f'REPLICAS: {REPLICAS}')

# not using metadata (only image, for now)
def read_labeled_tfrecord(example, __return_only_label):
    LABELED_TFREC_FORMAT = {
        "imagem": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        # 'image_name': tf.io.FixedLenFeature([], tf.string),
        'retinopatia' : tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['imagem'])
    label = tf.cast(example['retinopatia'], tf.int32)
    # name = example['image_name']

    # return image, label, name
    return image, label


def compute_class_weights(filenames):
    """
    Inverse-class-frequency weights (sklearn's standard `n_samples /
    (n_classes * n_samples_per_class)` formula), computed ONLY from the
    TRAINING TFRecord shards passed in — never from the test split, so this
    cannot leak test information into training. Reads just the integer
    label field (not the image) for speed.

    Opt-in via --use_class_weights (see experiments/MODEL_OPTIMIZATION.md
    for the rationale: FGADR referable-DR binarization is ~83%/17%
    imbalanced, which literature consistently identifies as a driver of
    depressed specificity in imbalanced binary medical classification).
    """
    label_only_format = {'retinopatia': tf.io.FixedLenFeature([], tf.int64)}
    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    ds = ds.map(lambda ex: tf.io.parse_single_example(ex, label_only_format)['retinopatia'])
    labels = [int(l.numpy()) for l in ds]
    n_total = len(labels)
    n_pos = sum(labels)
    n_neg = n_total - n_pos
    if n_total == 0 or n_pos == 0 or n_neg == 0:
        print(f'WARNING: cannot compute class weights (n_total={n_total}, n_pos={n_pos}, n_neg={n_neg}) '
              f'— falling back to unweighted loss')
        return None
    weight_for_0 = n_total / (2.0 * n_neg)
    weight_for_1 = n_total / (2.0 * n_pos)
    print(f'Class distribution in training split: class 0 (non-referable)={n_neg}, '
          f'class 1 (referable)={n_pos}, total={n_total}')
    print(f'Computed class weights: {{0: {weight_for_0:.4f}, 1: {weight_for_1:.4f}}}')
    return {0: weight_for_0, 1: weight_for_1}


def decode_image(image_data):
    # create-tfrecord.py encodes images as PNG (cv2.imencode('.png', ...)), so
    # decode as PNG here to match — decode_jpeg on PNG bytes is incorrect.
    image = tf.image.decode_png(image_data, channels=3)
    # Keep raw [0, 255] float range — EfficientNetV2's Keras Application
    # bundles its own Rescaling/Normalization as the first model layer, so
    # tf.keras.applications.efficientnet_v2.preprocess_input is a no-op by
    # design and pixels must NOT be divided by 255 here (see build_model()).
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    return image

# count # of images in files.. (embedded in file name)
def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
         for filename in filenames]
    return np.sum(n)

def load_dataset(filenames, labeled=True, ordered=False, return_only_label=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.experimental.AUTOTUNE) # automatically interleaves reads from multiple files
    dataset = dataset.cache()
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(lambda example: read_labeled_tfrecord(example, __return_only_label=return_only_label))
    # returns a dataset of (image, labels) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset


# GEOMETRIC-ONLY augmentation (random flip/rotation/translation). Applied
# identically regardless of which of the 576 preprocessing filters produced
# the image, so it is orthogonal to (does not confound) the photometric
# preprocessing comparison the campaign exists to make. Deliberately does
# NOT include any brightness/contrast/color/saturation augmentation, which
# WOULD confound that comparison by re-perturbing exactly the properties
# the filters manipulate — see experiments/MODEL_OPTIMIZATION_AUDIT.md.
# Built lazily (only if --augment is set) since it allocates Keras layers.
_augmentation_layer = None


def get_augmentation_layer():
    global _augmentation_layer
    if _augmentation_layer is None:
        # dtype='float32' pinned explicitly: Keras preprocessing layers
        # (RandomFlip/RandomRotation/RandomTranslation) only support
        # uint8/int32/int64/float16/float32/float64 internally, NOT
        # bfloat16 — under a 'mixed_bfloat16' global policy they would
        # otherwise auto-cast their input to bfloat16 and crash (confirmed
        # empirically). This matches standard mixed-precision guidance:
        # keep data augmentation/preprocessing in float32, only the
        # compute-heavy backbone layers benefit from reduced precision.
        _augmentation_layer = tf.keras.Sequential([
            tf.keras.layers.RandomFlip(mode='horizontal', dtype='float32'),
            tf.keras.layers.RandomRotation(factor=1.0, dtype='float32'),  # full 360 deg — fundus images have no fixed "up"
            tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05, dtype='float32'),
        ], name='geometric_augmentation')
    return _augmentation_layer


def get_training_dataset(filenames, _return_only_label=False, augment=False):
    dataset = load_dataset(filenames, labeled=True, return_only_label=_return_only_label)
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE*REPLICAS)
    if augment:
        aug = get_augmentation_layer()
        dataset = dataset.map(lambda img, label: (aug(img, training=True), label),
                               num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset


def build_metrics():
    return [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.AUC(name='AUC'),
        tf.keras.metrics.SensitivityAtSpecificity(0.95),
        tf.keras.metrics.SpecificityAtSensitivity(0.95),
        tf.keras.metrics.TruePositives(),
        tf.keras.metrics.TrueNegatives(),
        tf.keras.metrics.FalsePositives(),
        tf.keras.metrics.FalseNegatives()
    ]


def build_model():
    """
    Backbone EfficientNetV2S (ImageNet) + custom head replicando a topologia
    de Kao & Lin (2024): GAP -> Dropout(0.5) -> Dense(512, elu) ->
    Dropout(0.5) -> Dense(1, sigmoid).

    `base` é chamado explicitamente como submodelo aninhado (`base(inp)`,
    além de já receber `input_tensor=inp`) para preservar a estrutura que
    utils_gradcam.py espera (um layer `tf.keras.Model` aninhado dentro do
    modelo final, usado para localizar o backbone e sua última camada
    convolucional).
    """
    inp = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3))

    base = applications.EfficientNetV2S(weights='imagenet', include_top=False, input_tensor=inp)
    x = base(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='elu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inp, outputs=x)
    return model, base


def apply_finetune_depth(base, depth: str, partial_fraction: float):
    """
    'full' (default, pre-audit behavior): unfreeze the entire backbone.
    'partial': freeze the first `partial_fraction` of backbone layers BY
    INDEX (input side = most generic/early features, per discriminative
    fine-tuning literature — see experiments/MODEL_OPTIMIZATION_AUDIT.md),
    unfreeze only the rest. Index-based rather than layer-name-based so it
    doesn't depend on EfficientNetV2S's internal block naming.
    """
    base.trainable = True
    if depth == 'partial':
        n_layers = len(base.layers)
        n_frozen = int(n_layers * partial_fraction)
        for layer in base.layers[:n_frozen]:
            layer.trainable = False
        print(f'#### Partial fine-tune: {n_frozen}/{n_layers} backbone layers frozen '
              f'(fraction={partial_fraction})')


def build_optimizer(name: str, learning_rate, weight_decay: float, momentum: float, clipnorm,
                     use_ema: bool = False, ema_momentum: float = 0.99):
    """
    Optimizer factory (see experiments/MODEL_OPTIMIZATION_AUDIT.md for the
    literature justification of each choice). RMSProp intentionally not
    offered: it is the ORIGINAL EfficientNetV2 paper's choice for training
    FROM SCRATCH on ImageNet at large batch size — a different regime from
    fine-tuning a pretrained backbone on ~1300 medical images, with no
    literature support found for the latter case.
    """
    kwargs = {}
    if clipnorm is not None:
        kwargs['clipnorm'] = clipnorm
    if use_ema:
        kwargs['use_ema'] = True
        kwargs['ema_momentum'] = ema_momentum
    if name == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=learning_rate, **kwargs)
    elif name == 'adamw':
        return tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay, **kwargs)
    elif name == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def build_phase2_lr_schedule(scheduler: str, base_lr: float, warmup_steps_frac: float,
                             steps_per_epoch: int, epochs: int):
    """
    'reduce_on_plateau' (default, pre-audit behavior): returns base_lr
    unchanged — the ReduceLROnPlateau CALLBACK (added separately, reads
    val_loss) does the scheduling.
    'cosine_warmup': linear warmup then cosine decay to 0, purely a
    function of step count — does NOT read val_loss. This matters given
    this pipeline's known caveat that 'val_loss' is actually computed on
    the TEST split (see documentation/METHODOLOGICAL_NOTES.md item 1): a
    schedule that never looks at that signal cannot be indirectly shaped by
    it, which is a strictly cleaner methodological position for a schedule
    (as opposed to early stopping / restore_best_weights, which
    unavoidably still need some stopping signal and are left unchanged).
    """
    if scheduler == 'reduce_on_plateau':
        return base_lr
    elif scheduler == 'cosine_warmup':
        total_steps = steps_per_epoch * epochs
        warmup_steps = max(1, int(total_steps * warmup_steps_frac))
        return tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=base_lr, decay_steps=max(1, total_steps - warmup_steps),
            warmup_target=base_lr, warmup_steps=warmup_steps,
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler}")


# for others investigations we store all the history
histories = []

# these will be split in folds
num_total_train_files = len(tf.io.gfile.glob(TFREC_DIR + '/train*.tfrec'))
num_total_valid_files = len(tf.io.gfile.glob(TFREC_DIR + '/test*.tfrec'))

print('#### Image Size %i, batch_size %i'%
      (IMG_SIZES, BATCH_SIZE*REPLICAS))
print('#### Phase 1 (warm-up) epochs: %i, Phase 2 (fine-tuning) epochs: %i' % (PHASE1_EPOCHS, PHASE2_EPOCHS))

# CREATE TRAIN AND VALIDATION SUBSETS
TRAINING_FILENAMES = tf.io.gfile.glob(TFREC_DIR + '/train*.tfrec')
VALID_FILENAMES = tf.io.gfile.glob(TFREC_DIR + '/test*.tfrec')
print('Train TFRecord files', len(TRAINING_FILENAMES))
print('Train TFRecord files', len(VALID_FILENAMES))

if SHOW_FILES:
    print('Number of training images', count_data_items(TRAINING_FILENAMES))
    print('Number of validation images', count_data_items(VALID_FILENAMES))

# Default ON as of the HPO run of 2026-07-09 (see --use_class_weights).
# Computed from TRAINING_FILENAMES only, never VALID_FILENAMES (test split).
CLASS_WEIGHTS = compute_class_weights(TRAINING_FILENAMES) if USE_CLASS_WEIGHTS else None

K.clear_session()

print('#### Model number', exec)
with strategy.scope():
    model, base = build_model()

csv_logger_phase1 = tf.keras.callbacks.CSVLogger(results + '/' + dataset + '-%i-phase1_warmup.csv' % exec)
csv_logger_phase2 = tf.keras.callbacks.CSVLogger(results + '/' + dataset + '-%i-phase2_finetune.csv' % exec)

tStart = time.time()

# --- Fase 1: warm-up do topo (backbone congelado) ---
base.trainable = False
model.summary()

with strategy.scope():
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE1_LR),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=build_metrics(),
    )

print('=' * 80)
print(f'FASE 1: warm-up do topo (backbone congelado), {PHASE1_EPOCHS} épocas, lr={PHASE1_LR}')
print('=' * 80)
history_phase1 = model.fit(
    get_training_dataset(TRAINING_FILENAMES, augment=args.augment),
    epochs=PHASE1_EPOCHS,
    callbacks=[csv_logger_phase1],
    validation_data=get_training_dataset(VALID_FILENAMES),
    verbose=VERBOSE,
    class_weight=CLASS_WEIGHTS,
)
histories.append(history_phase1)

# --- Fase 2: fine-tuning (profundidade controlada por --finetune_depth) ---
apply_finetune_depth(base, args.finetune_depth, args.partial_freeze_fraction)
model.summary()

# count_data_items() (defined above) expects a "-<digits>." filename
# pattern that does NOT match create-tfrecord.py's actual shard naming
# ("..._cnt<N>.tfrec", no hyphen) — it has always been dead code (only
# reachable behind SHOW_FILES=False). Count directly from the real pattern
# instead of relying on it, purely to size the cosine-warmup schedule
# (an approximate step count is fine here; wrong by a few steps has no
# material effect on a cosine decay curve).
def _count_examples_from_shard_names(filenames):
    total = 0
    for fname in filenames:
        m = re.search(r"_cnt(\d+)\.tfrec$", fname)
        if m:
            total += int(m.group(1))
    return total


_n_train_examples = _count_examples_from_shard_names(TRAINING_FILENAMES) or 1
steps_per_epoch = max(1, _n_train_examples // (BATCH_SIZE * REPLICAS))
phase2_lr = build_phase2_lr_schedule(
    args.phase2_scheduler, PHASE2_LR, args.warmup_steps_frac, steps_per_epoch, PHASE2_EPOCHS
)
phase2_optimizer = build_optimizer(args.optimizer, phase2_lr, args.weight_decay, args.momentum, args.clipnorm,
                                    use_ema=args.use_ema, ema_momentum=args.ema_momentum)

with strategy.scope():
    model.compile(
        optimizer=phase2_optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=build_metrics(),
    )

phase2_callbacks = [csv_logger_phase2]
if args.phase2_scheduler == 'reduce_on_plateau':
    # Only meaningful when the LR itself is a plain float (not already a
    # step-based schedule) — cosine_warmup manages LR internally instead.
    phase2_callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE, verbose=1,
    ))
# restore_best_weights=True restores the best-val_loss weights in-memory at
# the end of training regardless of whether training stopped early or ran
# the full PHASE2_EPOCHS — this is what guarantees the final model.save()
# below persists the best epoch, not simply the last one. verbose=1 makes
# this restoration (and any early stop) visible in the training log instead
# of happening silently — needed to audit it across 180 unattended runs.
phase2_callbacks.append(tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1,
))

print('=' * 80)
print(f'FASE 2: fine-tuning ({args.finetune_depth}), até {PHASE2_EPOCHS} épocas, '
      f'otimizador={args.optimizer}, scheduler={args.phase2_scheduler}, lr inicial={PHASE2_LR}, '
      f'label_smoothing={LABEL_SMOOTHING}')
print('=' * 80)
history_phase2 = model.fit(
    get_training_dataset(TRAINING_FILENAMES, augment=args.augment),
    epochs=PHASE2_EPOCHS,
    callbacks=phase2_callbacks,
    validation_data=get_training_dataset(VALID_FILENAMES),
    verbose=VERBOSE,
    class_weight=CLASS_WEIGHTS,
)
histories.append(history_phase2)

tElapsed = round(time.time() - tStart, 1)

print(' ')
print('Time (sec) elapsed: ', tElapsed)
print('...')

# EarlyStopping(restore_best_weights=True) already restored the best-val_loss
# weights in-memory (see above) — no separate checkpoint reload needed here.
#
# use_ema=True optimizers keep a SHADOW moving-average of the weights that is
# NOT what model.save() persists by default — finalize_variable_values()
# overwrites the live trainable variables with that shadow average in place,
# which is required for the saved checkpoint to actually reflect the EMA
# weights rather than the raw (non-averaged) final-step weights.
if args.use_ema:
    phase2_optimizer.finalize_variable_values(model.trainable_variables)
model.save(results +'/'+ dataset + '-%i.keras' %exec)

# A post-training ROC-curve/thresholds-table/PDF diagnostic (gathering up
# to 2000 valid images into a numpy array, model.predict() on them, sklearn
# roc_curve/auc, a thresholds CSV, and a matplotlib ROC PDF) used to run
# here, plus a final redundant re-evaluation on the full train/valid
# TFRecord datasets. Both removed: neither is ever read by anything
# downstream — script3_avalia.py (the pipeline's real, authoritative
# evaluation step) computes its own independent predictions/metrics from
# scratch and never opens this script's thresholds CSV or PDF; nothing else
# in the pipeline references them either. Confirmed via repeated real runs
# during infrastructure validation that this diagnostic tail is also an
# active crash source on this GPU/TF-2.13 combination — model.predict()
# and/or the two extra full-dataset model.evaluate() calls occasionally
# triggered a low-level SIGBUS/std::bad_function_call abort (exit code -7
# or -6, no Python traceback — a signal-level OS kill that a try/except
# cannot catch, since it terminates the process before the interpreter can
# handle it) AFTER model.save() had already completed successfully — i.e.
# it turned an already-fully-successful, fully-checkpointed training run
# into a spurious job FAILURE for the sake of artifacts nothing consumes.
print(f"Time elapsed: {tElapsed}s")
