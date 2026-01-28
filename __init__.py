"""
DAVIS-2017 Dataset Loader for FiftyOne

This module provides functions to download and load the DAVIS-2017 dataset
into FiftyOne. The dataset contains video sequences with dense segmentation
annotations for the semi-supervised challenge.
"""

import os
import numpy as np
import cv2
from typing import Tuple

import fiftyone as fo
import fiftyone.core.labels as fol

from davis import DAVIS


SPLIT_TO_PATH = {
    "train": "trainval",
    "validation": "trainval",
    "test-dev": "test-dev",
    "test-challenge": "test-challenge",
}
SPLIT_TO_DAVIS_SPLIT = lambda x: "val" if x == "validation" else x


def download_and_prepare(dataset_dir, split=None, **kwargs):
    """Downloads the dataset and prepares it for loading into FiftyOne.

    Args:
        dataset_dir: the directory in which to construct the dataset
        split (None): a specific split to download, if the dataset supports
            splits. The supported split values are: "train", "validation", "test-dev", "test-challenge"
        **kwargs: optional keyword arguments that your dataset can define to
            configure what/how the download is performed

    Returns:
        a tuple of
        - ``dataset_type``: None (indicates custom loader will be used)
        - ``num_samples``: the total number of downloaded samples for the
            dataset or split
        - ``classes``: None (not applicable for segmentation datasets)
    """
    import urllib.request
    import zipfile

    BASE_URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/"
    SPLIT_TO_ZIP = lambda x: f"DAVIS-2017-{x}-480p.zip"

    # Validate split
    if split is not None and split not in SPLIT_TO_PATH:
        raise ValueError(
            f"Invalid split '{split}'. Must be one of {SPLIT_TO_PATH.keys()}"
        )

    # Determine base directory for downloads
    # If dataset_dir ends with a split name, use parent directory for shared data
    base_dir = dataset_dir
    if split is not None:
        # Check if dataset_dir is split-specific (ends with split name)
        if dataset_dir.endswith(split):
            base_dir = os.path.dirname(dataset_dir)
    
    # Create directory structure
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    # Download and extract zip files
    # zips_to_download = [SPLIT_TO_ZIP(SPLIT_TO_PATH[split])] if split else [SPLIT_TO_ZIP(split) for split in SPLIT_TO_PATH.keys()]
    # for zip_filename in zips_to_download:
    # Map zip filenames to their corresponding splits
    if split:
        zips_to_download = [(SPLIT_TO_ZIP(SPLIT_TO_PATH[split]), split)]
    else:
        # Download all splits - map each zip to its split
        zips_to_download = []
        for fo_split, path_key in SPLIT_TO_PATH.items():
            zip_name = SPLIT_TO_ZIP(path_key)
            # Only add unique zips (train and validation share the same zip)
            if not any(z[0] == zip_name for z in zips_to_download):
                zips_to_download.append((zip_name, fo_split))
    
    for zip_filename, zip_split in zips_to_download:
        # Download to base directory (shared location)
        zip_path = os.path.join(base_dir, zip_filename)
        zip_url = BASE_URL + zip_filename

        # Download if not already present
        if not os.path.exists(zip_path):
            print(f"Downloading {zip_filename}...")
            try:
                urllib.request.urlretrieve(zip_url, zip_path)
                print(f"Downloaded {zip_filename}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download {zip_url}: {e}"
                ) from e
        else:
            print(f"{zip_filename} already exists, skipping download")

        # Determine extraction directory using the split for this zip
        extract_dir = os.path.join(base_dir, SPLIT_TO_PATH[zip_split])

        # Check if already extracted (look for JPEGImages directly in extract_dir)
        jpeg_images_dir = os.path.join(extract_dir, "JPEGImages")
        if not os.path.exists(jpeg_images_dir):
            print(f"Extracting {zip_filename}...")
            try:
                # Extract to a temporary location first
                temp_extract = os.path.join(extract_dir, "_temp_extract")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_extract)
                
                # The zip contains a DAVIS folder, move its contents up one level
                temp_davis = os.path.join(temp_extract, "DAVIS")
                if os.path.exists(temp_davis):
                    # Move contents of DAVIS folder to extract_dir
                    import shutil
                    for item in os.listdir(temp_davis):
                        src = os.path.join(temp_davis, item)
                        dst = os.path.join(extract_dir, item)
                        if os.path.exists(dst):
                            shutil.rmtree(dst) if os.path.isdir(dst) else os.remove(dst)
                        shutil.move(src, dst)
                    # Remove temporary extraction directory
                    shutil.rmtree(temp_extract)
                else:
                    # No DAVIS folder, move temp_extract contents directly
                    for item in os.listdir(temp_extract):
                        src = os.path.join(temp_extract, item)
                        dst = os.path.join(extract_dir, item)
                        if os.path.exists(dst):
                            shutil.rmtree(dst) if os.path.isdir(dst) else os.remove(dst)
                        shutil.move(src, dst)
                    os.rmdir(temp_extract)
                
                print(f"Extracted {zip_filename}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to extract {zip_path}: {e}"
                ) from e
        else:
            print(f"Data already extracted to {extract_dir}, skipping extraction")
    
    # Verify extraction
    num_samples = 0
    splits_to_count = [split] if split else SPLIT_TO_PATH.keys()
    for split in splits_to_count:
        imagesets_file = os.path.join(
            base_dir, SPLIT_TO_PATH[split], "ImageSets", "2017", f"{SPLIT_TO_DAVIS_SPLIT(split)}.txt"
        )
        if os.path.exists(imagesets_file):
            with open(imagesets_file, 'r') as f:
                sequences = [line.strip() for line in f if line.strip()]
                num_samples += len(sequences)
    
    # Return None to indicate custom loader will be used
    return None, num_samples, None


def load_dataset(dataset, dataset_dir, split=None, **kwargs):
    """Loads the dataset into the given FiftyOne dataset.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset` to which to import
        dataset_dir: the directory to which the dataset was downloaded
        split (None): a split to load. The supported values are
            ``("train", "validation", "test-dev", "test-challenge")``
        **kwargs: optional keyword arguments that your dataset can define to
            configure what/how the load is performed
    
    DAVIS-2017 structure:
    dataset_dir/
        split_dir/
            JPEGImages/480p/{sequence_name}/*.jpg  # Frame images
            Annotations/480p/{sequence_name}/*.png  # Segmentation masks
            ImageSets/2017/{split}.txt  # List of sequence names
    """
    # Validate split
    if split is not None and split not in SPLIT_TO_PATH:
        raise ValueError(
            f"Invalid split '{split}'. Must be one of {SPLIT_TO_PATH.keys()}"
        )

    # Get sequence names for each split
    splits_to_load = [split] if split else SPLIT_TO_PATH.keys()
    for split in splits_to_load:
        split_dir = os.path.join(dataset_dir, SPLIT_TO_PATH[split])
        if not os.path.exists(split_dir):
            raise ValueError(
                f"Dataset directory does not exist: {split_dir}. "
                f"Expected DAVIS data at this location. "
                f"Base dataset_dir (input): {dataset_dir}, Split: {split}"
            )

        davis_split = SPLIT_TO_DAVIS_SPLIT(split)
        davis_split_object = DAVIS(root=split_dir, subset=davis_split)

        format = kwargs.get("format", "image")
        if format == "image":
            _load_image_dataset(dataset, davis_split_object)
        elif format == "video":
            _load_video_dataset(dataset, davis_split_object)
        else:
            raise ValueError(f"Invalid format: {format}. Must be one of ['image', 'video']")


def _load_image_dataset(dataset: fo.Dataset, davis_split_object: DAVIS):
    """
    Load the dataset object into the given FiftyOne dataset
    as an image dataset, with sequence names as tags, and frame images as samples
    """
    for seq in davis_split_object.get_sequences():
        images, image_frame_numbers = davis_split_object.get_all_images(seq)
        masks, masks_void, mask_frame_numbers = davis_split_object.get_all_masks(seq)
        if "test" not in davis_split_object.subset:
            assert image_frame_numbers == mask_frame_numbers

        for img, mask, image_frame_number in zip(images, masks, image_frame_numbers):
            img = img.astype(np.uint8)
            mask = mask.astype(np.uint8)

            filepath = os.path.join(
                davis_split_object.root,
                "JPEGImages", "480p",
                seq, f"{image_frame_number}.jpg"
            )
            assert os.path.exists(filepath)

            detections = []

            num_classes = int(np.max(mask))
            for cc in range(num_classes):
                mask_cc = (mask == cc+1).astype(np.uint8)
                bounding_box = cv2.boundingRect(mask_cc)
                if np.sum(mask_cc) == 0:
                    rel_mask = None
                else:
                    rel_mask = mask_cc[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2]]
                
                normalized_bounding_box = [
                    bounding_box[0]/img.shape[1], bounding_box[1]/img.shape[0],
                    bounding_box[2]/img.shape[1], bounding_box[3]/img.shape[0]
                ]
                detections.append(fo.Detection(
                    bounding_box=normalized_bounding_box,
                    mask=rel_mask,
                    label=seq+str(cc),
                ))
        
            # Create sample with metadata
            sample = fo.Sample(
                filepath=str(filepath),
                tags=[davis_split_object.subset, seq],
                frame_number=image_frame_number,
                ground_truth=fo.Detections(detections=detections),
            )
            
            dataset.add_sample(sample)


def _load_video_dataset(dataset: fo.Dataset, davis_split_object: DAVIS):
    """
    Load the dataset object into the given FiftyOne dataset
    as a video dataset, with sequences as samples, and frame images as frames
    """
    raise NotImplementedError("Video dataset loading not implemented yet")
