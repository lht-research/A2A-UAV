from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadPanoramaTileFromFile(BaseTransform):

    def __init__(self, to_float32: bool = False, cache_last_panorama: bool = True) -> None:
        self.to_float32 = to_float32
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

    def transform(self, results: dict) -> Optional[dict]:
        img_path = str(results["img_path"])
        img_exists = Path(img_path).exists()
        use_virtual_tile = results.get("virtual_tile") or (
            not img_exists
            and results.get("source_file_name")
            and "tile_x" in results
            and "tile_y" in results
            and "tile_size" in results
        )
        if use_virtual_tile:
            source_path = str(results.get("source_file_name") or img_path)
            full = self._load_full_panorama(source_path)
            tile_x = int(results["tile_x"])
            tile_y = int(results["tile_y"])
            tile_size = int(results["tile_size"])
            array = full[tile_y : tile_y + tile_size, tile_x : tile_x + tile_size]
            if array.shape[0] != tile_size or array.shape[1] != tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=array.dtype)
                padded[: array.shape[0], : array.shape[1]] = array
                array = padded
        else:
            array = self._load_image(img_path)

        if self.to_float32:
            array = array.astype(np.float32)


        array = array[:, :, ::-1].copy()
        results["img"] = array
        results["img_shape"] = array.shape[:2]
        results["ori_shape"] = array.shape[:2]
        results["height"] = array.shape[0]
        results["width"] = array.shape[1]
        results["img_path"] = img_path
        results["img_fields"] = ["img"]
        results["scale_factor"] = (1.0, 1.0)
        return results
