# -*- coding: utf-8 -*-
"""
create-tfrecord.py — Geração de TFRecords a partir do dataset filtrado (por filtro)

Objetivo: Converter datasets_filtrados/{filtro}/{train,test}/*.png em TFRecords
          consumíveis por dr_hcpa_v2_2024.py (train*.tfrec / test*.tfrec).

Entrada: datasets_filtrados/{filtro}/{train,test}/*.png (Script 1), split.json,
         data/Seg-set/DR_Seg_Grading_Label.csv (rótulos ICDR, binarizados em
         referable DR: grade >= threshold -> 1).

Saída: {tfrecords_filtrados}/{filtro}/train_shardNN_cntXX.tfrec
       {tfrecords_filtrados}/{filtro}/test_shardNN_cntXX.tfrec

Granularidade: Uma vez por filtro (reutilizado pelas 10 repetições de treino).

Execução: CPU-only.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import argparse
import glob

import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils_common import setup_logging, get_paths, validate_paths, load_label_map, filter_lock


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _string_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(img_bytes, name, label):
    feature = {
        'imagem': _bytes_feature(img_bytes),
        'image_name': _string_feature(name),
        'retinopatia': _int64_feature(label),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def create_tfrecords_for_split(image_paths, split_name, output_dir, label_map, size_limit, img_size, logger):
    """Write one split (train/test) of a filter's images into sharded TFRecords."""
    import cv2

    total_imgs = len(image_paths)
    if total_imgs == 0:
        logger.warning(f"No images to process for split '{split_name}'")
        return 0

    num_shards = total_imgs // size_limit + int(total_imgs % size_limit != 0)
    logger.info(f"  {split_name}: {total_imgs} images -> {num_shards} shard(s)")

    written = 0
    skipped_no_label = 0
    for shard_id in range(num_shards):
        start_idx = shard_id * size_limit
        end_idx = min((shard_id + 1) * size_limit, total_imgs)
        batch = image_paths[start_idx:end_idx]

        tfrec_path = output_dir / f"{split_name}_shard{shard_id:02d}_cnt{len(batch)}.tfrec"
        with tf.io.TFRecordWriter(str(tfrec_path)) as writer:
            for img_path in batch:
                img_name = os.path.basename(img_path)
                if img_name not in label_map:
                    logger.warning(f"Label not found for {img_name}, skipping")
                    skipped_no_label += 1
                    continue

                img_cv = cv2.imread(img_path)
                if img_cv is None:
                    logger.warning(f"Could not read {img_path}, skipping")
                    continue

                # dr_hcpa_v2_2024.py's decode_image() reshapes to a fixed
                # (img_size, img_size, 3) tensor — images must already be at
                # that resolution when written (source images are 1280x1280).
                if img_cv.shape[:2] != (img_size, img_size):
                    img_cv = cv2.resize(img_cv, (img_size, img_size), interpolation=cv2.INTER_AREA)

                img_bytes = cv2.imencode('.png', img_cv)[1].tobytes()
                label = label_map[img_name]

                writer.write(serialize_example(img_bytes, img_name, label))
                written += 1

    if skipped_no_label:
        logger.warning(f"  {split_name}: {skipped_no_label} image(s) skipped (no label found)")

    return written


def create_tfrecords_for_filter(filter_name: str, paths: dict, size_limit: int,
                                binarize_threshold: int, img_size: int, logger, force: bool = False) -> bool:
    """
    Acquire the per-filter lock, then convert the filtered dataset to
    TFRecords. The lock serializes any concurrent invocation for the SAME
    filter (e.g. a stray script2_treina.py auto-generation call racing with
    the orchestrator's dedicated TFRecord job), preventing two processes
    from writing to the same shard files concurrently.
    """
    with filter_lock(paths, filter_name, logger):
        return _create_tfrecords_for_filter_locked(
            filter_name, paths, size_limit, binarize_threshold, img_size, logger, force
        )


def _create_tfrecords_for_filter_locked(filter_name: str, paths: dict, size_limit: int,
                                        binarize_threshold: int, img_size: int, logger, force: bool = False) -> bool:
    filtered_dataset_dir = paths["datasets_filtrados"] / filter_name
    output_dir = paths["tfrecords_filtrados"] / filter_name

    if not filtered_dataset_dir.exists():
        logger.error(f"Filtered dataset not found: {filtered_dataset_dir} (run script1 first)")
        return False

    if output_dir.exists():
        existing = list(output_dir.glob("train*.tfrec")) + list(output_dir.glob("test*.tfrec"))
        if existing and not force:
            logger.info(f"TFRecords already exist in {output_dir} ({len(existing)} files) — skipping")
            return True
        if force:
            import shutil
            shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    label_map = load_label_map(paths["labels_csv"], threshold=binarize_threshold)
    logger.info(f"Loaded {len(label_map)} labels (binarized: grade >= {binarize_threshold} -> referable DR)")

    total_written = 0
    for split_name in ["train", "test"]:
        split_dir = filtered_dataset_dir / split_name
        if not split_dir.exists():
            logger.error(f"Split directory not found: {split_dir}")
            return False

        image_paths = sorted(glob.glob(str(split_dir / "*.png")))
        written = create_tfrecords_for_split(image_paths, split_name, output_dir, label_map, size_limit, img_size, logger)
        total_written += written

    if total_written == 0:
        logger.error("No TFRecord examples were written — aborting")
        return False

    logger.info(f"✓ Wrote {total_written} examples to {output_dir}")
    return True


def parse_arguments():
    parser = argparse.ArgumentParser(description='Create per-filter TFRecords for the XAI preprocessing pipeline.')
    parser.add_argument('--filtro', type=str, required=True, help='Filter name (must match a datasets_filtrados/{filtro} directory)')
    parser.add_argument('--base-ssd', type=str, default=None, help='Base SSD path')
    parser.add_argument('--base-home', type=str, default=None, help='Base home path')
    parser.add_argument('--size', type=int, default=100, help='Number of images per TFRecord shard')
    parser.add_argument('--img-size', type=int, default=299,
                        help='Resize images to (img_size, img_size) before encoding — must match '
                             'dr_hcpa_v2_2024.py --img_sizes (default 299)')
    parser.add_argument('--binarize-threshold', type=int, default=2,
                        help='ICDR grade >= threshold is labeled referable DR (1); default 2')
    parser.add_argument('--force', action='store_true', help='Regenerate TFRecords even if they already exist')
    return parser.parse_args()


def main():
    args = parse_arguments()
    logger = setup_logging(f"create_tfrecord_{args.filtro}")

    paths = get_paths(args.base_ssd, args.base_home)
    if not validate_paths(paths, logger):
        logger.error("Path validation failed!")
        sys.exit(1)

    success = create_tfrecords_for_filter(
        args.filtro, paths, args.size, args.binarize_threshold, args.img_size, logger, force=args.force
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
