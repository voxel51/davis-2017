"""
Run this script in the foe environment
"""

import fiftyone as fo
import fiftyone.core.storage as fos
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "https://github.com/voxel51/davis-2017",
    split="validation",
    drop_existing_dataset=True,
    format="image",
)

cloud_dir = "gs://voxel51-test/DAVIS-2017/"
rel_dir = "/".join(dataset.first()['filepath'].split('/')[:-2])

fos.upload_media(
    dataset,
    cloud_dir,
    rel_dir=rel_dir,
    update_filepaths=False,
    overwrite=False,
    progress=True,
)
