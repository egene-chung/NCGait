# CASIA-B and CASIA-B\*
Download URL: http://www.cbsr.ia.ac.cn/GaitDatasetB-silh.zip
- Original
    ```
    CASIA-B
        001 (subject)
            bg-01 (type)
                000 (view)
                    001-bg-01-000-001.png (frame)
                    001-bg-01-000-002.png (frame)
                    ......
                ......
            ......
        ......
    ```
- Run `python datasets/pretreatment_CASIA-B.py --input_path CASIA-B --output_path CASIA-B-pkl`
- Processed
    ```
    CASIA-B-pkl
        001 (subject)
            bg-01 (type)
                    000 (view)
                        000.pkl (contains all frames)
                ......
            ......
        ......
    ```
    
# CASIA-B\*
## Introduction
CASIA-B\* is a re-segmented version of CASIA-B processed by Liang et al. It replaces the original background-subtraction pipeline, which can produce noisy silhouettes, with updated pedestrian tracking and segmentation methods. The resulting dataset contains cropped RGB images, binary silhouettes, bounding-box height-to-width ratios, and aligned silhouettes. Please refer to [GaitEdge](../../configs/gaitedge/README.md) for more details. To request this subset, follow the instructions at [http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp] and specify Dataset B* in the email subject.

## Data structure
```
casiab-128-end2end/
    001 (subject)
        bg-01 (type)
                000 (view)
                    000-aligned-sils.pkl (aligned sils, nx64x44)
                    000-ratios.pkl (aspect ratio of bounding boxes, n)
                    000-rgbs.pkl (cropped RGB images, nx3x128x128)
                    000-sils.pkl (binary silhouettes, nx128x128)
            ......
        ......
    ......
```

## How to use
By default, it loads all file directory information like other datasets before training starts. If you need to use some of these data separately, such as `aligned-sils`, then you can use the `data_in_use` parameter in `data_cfg` lexicographically, *i.e.* `data_in_use: [true, false, false, false]`.
