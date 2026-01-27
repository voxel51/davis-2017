# davis-2017
FiftyOne Dataset with the Densely Annotated Video Segmentation semi-supervised challenge data (https://davischallenge.org/davis2017/code.html)

## Details

- Original Website: https://davischallenge.org/davis2017/code.html
- Original Source code for evaluation: https://github.com/davisvideochallenge/davis2017-evaluation
- Citation
```
@article{Pont-Tuset_arXiv_2017,
  author = {Jordi Pont-Tuset and Federico Perazzi and Sergi Caelles and Pablo Arbel\'aez and Alexander Sorkine-Hornung and Luc {Van Gool}},
  title = {The 2017 DAVIS Challenge on Video Object Segmentation},
  journal = {arXiv:1704.00675},
  year = {2017}
}
```
- Tags: image, detection, segmentation
- Supported split: train, validation, test-dev

## Example Usage

```
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("https://github.com/voxel51/davis-2017")

session = fo.launch_app(dataset)
```

## Statistics

| Split | Sequences | Total Samples | Annotated Samples |
|-------|-----------|---------|-------------------|
| train | 59 | 4,209 | 4,209 |
| validation | 30 | 1,999 | 1,999 |
| test-dev | 30 | 2,086 | 30 |
| test-challenge | 30 | 2,180 | 30 |

## Visualize

Each image is tagged with its split, and with its sequence name -- images from a "sequence" are from a single video clip.
Sequences do not overlap across splits.

The bounding box and segmentation mask is stored in the `ground_truth` field

![DAVIS Sample Visualization](assets/davis_grid.png)