from __future__ import annotations

import os.path as osp
from pathlib import Path
from typing import Optional

import mmcv
import numpy as np
from PIL import Image
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadPanoramaTileFromFile:

    def __init__(self, to_float32: bool = False, color_type: str = "color", cache_last_panorama: bool = True):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.cache_last_panorama = cache_last_panorama
        self._cached_path: Optional[str] = None
        self._cached_array: Optional[np.ndarray] = None

    def _load_full_panorama(self, path: str) -> np.ndarray:
        if self.cache_last_panorama and path == self._cached_path and self._cached_array is not None:
            return self._cached_array
        image = Image.open(path).convert("RGB")
        array = np.asarray(image)
        if self.cache_last_panorama:
            self._cached_path = path
            self._cached_array = array
        return array

    def _load_image(self, path: str) -> np.ndarray:
        image = Image.open(path).convert("RGB")
        return np.asarray(image)

    def __call__(self, results: dict) -> dict:
        img_info = results["img_info"]
        img_prefix = results.get("img_prefix")
        filename = img_info.get("filename") or img_info.get("file_name")
        if img_prefix is not None and filename and not osp.isabs(str(filename)):
            img_path = osp.join(img_prefix, str(filename))
        else:
            img_path = str(filename)

        use_virtual_tile = img_info.get("virtual_tile") or (
            img_info.get("source_file_name")
            and "tile_x" in img_info
            and "tile_y" in img_info
            and "tile_size" in img_info
            and not Path(img_path).exists()
        )
        if use_virtual_tile:
            source_path = str(img_info.get("source_file_name") or img_path)
            full = self._load_full_panorama(source_path)
            tile_x = int(img_info["tile_x"])
            tile_y = int(img_info["tile_y"])
            tile_size = int(img_info["tile_size"])
            img = full[tile_y:tile_y + tile_size, tile_x:tile_x + tile_size]
            if img.shape[0] != tile_size or img.shape[1] != tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=img.dtype)
                padded[:img.shape[0], :img.shape[1]] = img
                img = padded
        else:
            img = self._load_image(img_path)

        if self.color_type == "color":
            img = img[:, :, ::-1].copy()
        else:
            img = mmcv.imconvert(img, "rgb", self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)

        results["filename"] = img_path
        results["ori_filename"] = str(filename)
        results["img"] = img
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        results["img_fields"] = ["img"]
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(to_float32={self.to_float32}, color_type={self.color_type})"
