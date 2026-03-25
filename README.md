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

The bounding box and segmentation mask is stored in the `ground_truth` field.

Below is an example of the dataset loaded in `image` format. Coming soon: `video`.

![DAVIS Sample Visualization](assets/davis_grid.png)
