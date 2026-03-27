# davis-2017

<div align="center">

[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/fiftyone-community)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-purple?style=flat&logo=huggingface)](https://huggingface.co/Voxel51)
[![Voxel51 Blog](https://img.shields.io/badge/Voxel51_Blog-ff6d04?style=flat)](https://voxel51.com/blog)
[![Newsletter](https://img.shields.io/badge/Newsletter-BE5B25?logo=mail.ru&logoColor=white)](https://share.hsforms.com/1zpJ60ggaQtOoVeBqIZdaaA2ykyk)
[![LinkedIn](https://img.shields.io/badge/In-white?style=flat&label=Linked&labelColor=blue)](https://www.linkedin.com/company/voxel51)
[![Twitter](https://img.shields.io/badge/Twitter-000000?logo=x&logoColor=white)](https://x.com/voxel51)
[![Medium](https://img.shields.io/badge/Medium-12100E?logo=medium&logoColor=white)](https://medium.com/voxel51)

</div>

A [FiftyOne remote zoo dataset](https://docs.voxel51.com/dataset_zoo/remote.html) for **DAVIS 2017** — the Densely Annotated VIdeo Segmentation challenge release used for the semi-supervised video object segmentation track. Data are organized as per-frame samples (or as generated video clips) with instance-level masks and boxes. Official challenge hub: [davischallenge.org/davis2017/code.html](https://davischallenge.org/davis2017/code.html).

## Source, citation, and license

- **Challenge / download page:** [https://davischallenge.org/davis2017/code.html](https://davischallenge.org/davis2017/code.html)
- **Evaluation code (reference implementation):** [https://github.com/davisvideochallenge/davis2017-evaluation](https://github.com/davisvideochallenge/davis2017-evaluation)
- **License:** same terms as the DAVIS release — see [LICENSE in davis2017-evaluation](https://github.com/davisvideochallenge/davis2017-evaluation/blob/master/LICENSE)
- **Citation:**

```
@article{Pont-Tuset_arXiv_2017,
  author = {Jordi Pont-Tuset and Federico Perazzi and Sergi Caelles and Pablo Arbel\'aez and Alexander Sorkine-Hornung and Luc {Van Gool}},
  title = {The 2017 DAVIS Challenge on Video Object Segmentation},
  journal = {arXiv:1704.00675},
  year = {2017}
}
```


## Quick start

Install FiftyOne, then load the dataset by GitHub URL (downloads and caches automatically when needed):

```bash
pip install fiftyone
```

```python
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "https://github.com/voxel51/davis-2017",
    split="validation",  # optional, for a specific split
    max_samples=500,  # optional, for quick exploration
    format="image",  # "image" (default) or "video"
)

session = fo.launch_app(dataset)

# Dynamically Grouped Dataset
grouped_view = dataset.group_by("sequence_id", order_by="frame_number")

# Explore a single sequence
seq_view = dataset.match_tags("blackswan")
```

Notes:
- Supported `split` values: `train`, `validation`, `test-dev`, `test-challenge`.
- Resolution: 480p (matches the `DAVIS-2017-*-480p.zip` archives fetched from the official host)
- Image mode creates one sample per frame. Video mode stitches each sequence into an `.mp4` and creates a local copy under the dataset tree. Attaches `ground_truth` on frames where annotations exist.

## Statistics

| Split | Sequences | Total Samples | Annotated Samples |
|-------|-----------|---------------|-------------------|
| train | 59 | 4,209 | 4,209 |
| validation | 30 | 1,999 | 1,999 |
| test-dev | 30 | 2,086 | 30 |
| test-challenge | 30 | 2,180 | 30 |

Training and validation splits are fully annotated. For **test-dev** and **test-challenge**, only the first frames carry dense masks (hence the small “annotated” counts in the table); remaining frames appear without `ground_truth` detections.

## Sample fields

Each sample includes:

- **`filepath`** — JPEG for that frame (image / group modes) or generated video path (`format="video"`).
- **`tags`** —
  - DAVIS subset name (`train`, `val`, `test-dev`, or `test-challenge`; note the zoo split is `validation` but the tag is `val`)
  - DAVIS the sequence name
- **`sequence_id`** — DAVIS sequence (clip) name. Note: Sequences do not overlap accross splits.
- **`frame_number`** — frame id within the sequence (image / group modes; frame index in video mode).
- **`ground_truth`** — [`fo.Detections`](https://docs.voxel51.com/api/fiftyone.core.labels.html#fiftyone.core.labels.Detections) with one [`Detection`](https://docs.voxel51.com/api/fiftyone.core.labels.html#fiftyone.core.labels.Detection) per object: normalized bounding box, instance mask (`mask`), and label string `{sequence_id}{object_index}`.

## Visualization

Example grid with the dataset loaded in **image** format:

![DAVIS Sample Visualization](assets/davis_grid.png)
