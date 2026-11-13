from __future__ import annotations

import math
import os.path as osp
from typing import List, Union

from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class UavCocoTileDataset(CocoDataset):

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        img_info = raw_data_info["raw_img_info"]
        ann_info = raw_data_info["raw_ann_info"]

        data_info = {}
        if img_info.get("virtual_tile") and img_info.get("source_file_name"):
            img_path = str(img_info["source_file_name"])
        else:
            file_name = str(img_info["file_name"])
            if osp.isabs(file_name):
                img_path = file_name
            else:
                img_path = osp.join(self.data_prefix["img"], file_name)

        if self.data_prefix.get("seg", None):
            seg_map_path = osp.join(
                self.data_prefix["seg"],
                str(img_info["file_name"]).rsplit(".", 1)[0] + self.seg_map_suffix,
            )
        else:
            seg_map_path = None

        data_info["img_path"] = img_path
        data_info["img_id"] = img_info["id"]
        data_info["seg_map_path"] = seg_map_path
        data_info["height"] = int(img_info["height"])
        data_info["width"] = int(img_info["width"])

        for key in (
            "file_name",
            "source_file_name",
            "source_image_id",
            "sample_id",
            "recording_key",
            "condition",
            "scene",
            "session",
            "split",
            "tile_x",
            "tile_y",
            "tile_size",
            "source_width",
            "source_height",
            "virtual_tile",
        ):
            if key in img_info:
                data_info[key] = img_info[key]

        if self.return_classes:
            data_info["text"] = self.metainfo["classes"]
            data_info["caption_prompt"] = self.caption_prompt
            data_info["custom_entities"] = True

        instances = []
        for ann in ann_info:
            if ann.get("ignore", False):
                continue
            x1, y1, w, h = ann["bbox"]
            values = (x1, y1, w, h, ann.get("area", 0))
            if any(not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in values):
                continue
            x1 = float(x1)
            y1 = float(y1)
            w = float(w)
            h = float(h)
            inter_w = max(0, min(x1 + w, img_info["width"]) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info["height"]) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue
            if ann["category_id"] not in self.cat_ids:
                continue
            x2 = min(max(x1 + w, x1 + 1.0), float(img_info["width"]))
            y2 = min(max(y1 + h, y1 + 1.0), float(img_info["height"]))
            x1 = min(max(x1, 0.0), x2 - 1.0)
            y1 = min(max(y1, 0.0), y2 - 1.0)
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            instance = {
                "bbox": [x1, y1, x2, y2],
                "bbox_label": self.cat2label[ann["category_id"]],
                "ignore_flag": 1 if ann.get("iscrowd", False) else 0,
            }
            if ann.get("segmentation", None):
                instance["mask"] = ann["segmentation"]
            instances.append(instance)
        data_info["instances"] = instances
        return data_info
