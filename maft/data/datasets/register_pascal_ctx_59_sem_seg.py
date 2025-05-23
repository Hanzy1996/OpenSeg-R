"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 
"""

import os
from pathlib import Path

import numpy as np

from detectron2.data import DatasetCatalog, MetadataCatalog
from .load_sem_seg import load_sem_seg


from . import openseg_classes

PASCAL_CTX_59_CATEGORIES=openseg_classes.get_pascal_ctx_59_categories_with_prompt_eng()

PASCAL_CTX_59_COLORS = [k["color"] for k in PASCAL_CTX_59_CATEGORIES]

MetadataCatalog.get("openvocab_pascal_ctx59_sem_seg_train").set(
    stuff_colors=PASCAL_CTX_59_COLORS[:],
)

MetadataCatalog.get("openvocab_pascal_ctx59_sem_seg_val").set(
    stuff_colors=PASCAL_CTX_59_COLORS[:],
)

def _get_ctx59_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing, so all ids are shifted by 1.
    stuff_ids = [k["id"] for k in PASCAL_CTX_59_CATEGORIES]
    assert len(stuff_ids) == 59, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in PASCAL_CTX_59_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret


def register_all_ctx59(root):
    # root = os.path.join(root, "pascal_ctx_d2")
    meta = _get_ctx59_meta()
    for name, dirname in [("train", "training"), ("val", "val")]:
        image_dir = os.path.join(root, "JPEGImages")
        gt_dir = os.path.join(root, 'annotations_detectron2', "pc59_val")
        name = f"openvocab_pascal_ctx59_sem_seg_{name}" 
        meta['dataname'] = name
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg", meta = meta)
        )
        MetadataCatalog.get(name).set(
            stuff_classes=meta["stuff_classes"][:],
            thing_dataset_id_to_contiguous_id={},  # to make Mask2Former happy
            stuff_dataset_id_to_contiguous_id=meta["stuff_dataset_id_to_contiguous_id"],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            gt_ext="png",
            dataset_name = name,
        )
        
_root = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "VOCdevkit" / "VOC2010"
register_all_ctx59(_root)