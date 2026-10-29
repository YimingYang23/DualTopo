# SPDX-License-Identifier: Apache-2.0
# Portions copyright OpenDriveLab; see NOTICE.
# Modified for the DualTopo inference-only release: paper names and portable interfaces.
"""Inference image preprocessing and LaneSegNet-compatible prediction formatting."""

import os
import pickle
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from shapely.geometry import LineString
from pyquaternion import Quaternion


def fix_pts_interpolate(lane, n_points):
    """Interpolate a lane polyline to exactly n_points equally-spaced points."""
    ls = LineString(lane)
    distances = np.linspace(0, ls.length, n_points)
    return np.array([ls.interpolate(d).coords[0] for d in distances])


def to_tensor(data):
    """Convert numpy array or others to torch.Tensor."""
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, (list, tuple)):
        if len(data) > 0 and isinstance(data[0], np.ndarray):
            return torch.from_numpy(np.array(data))
        return torch.tensor(data)
    else:
        return torch.tensor(data)


def imread(filepath, color_type="unchanged"):
    """Read an image from file (replaces mmcv.imread)."""
    if color_type == "color" or color_type == 1:
        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    elif color_type == "grayscale" or color_type == 0:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read camera image: {filepath}")
    return img


def impad(img, *, shape=None, pad_val=0):
    """Pad an image to the given shape (replaces mmcv.impad).

    Pads on the **bottom and right** only, matching mmcv.impad behavior:
    padding = (0, 0, width, height) → copyMakeBorder(top=0, bottom=pad_h, left=0, right=pad_w).
    """
    if shape is not None:
        pad_h = max(shape[0] - img.shape[0], 0)
        pad_w = max(shape[1] - img.shape[1], 0)
    else:
        raise ValueError("shape must be specified")
    return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=pad_val)


def impad_to_multiple(img, divisor, *, pad_val=0):
    """Pad an image so each dimension is divisible by divisor (replaces mmcv.impad_to_multiple)."""
    pad_h = (divisor - img.shape[0] % divisor) % divisor
    pad_w = (divisor - img.shape[1] % divisor) % divisor
    if pad_h == 0 and pad_w == 0:
        return img
    if len(img.shape) == 3:
        return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=pad_val)
    else:
        return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=pad_val)


def imnormalize(img, mean, std, to_rgb=True):
    """Normalize an image with mean and std (replaces mmcv.imnormalize)."""
    img = img.copy().astype(np.float32)
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    img = (img - mean) / std
    return img


def imresize(img, size, return_scale=False, interpolation=cv2.INTER_LINEAR):
    """Resize image (replaces mmcv.imresize). size is (width, height)."""
    h, w = img.shape[:2]
    resized = cv2.resize(img, size, interpolation=interpolation)
    if return_scale:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return (resized, w_scale, h_scale)
    return resized


_PIPELINE_REGISTRY = {}


def register_pipeline(cls):
    """Decorator to register a pipeline transform class."""
    _PIPELINE_REGISTRY[cls.__name__] = cls
    return cls


class Compose:
    """Compose multiple pipeline transforms sequentially.

    Args:
        pipelines (list[dict]): Each dict must contain a 'type' key
            specifying the transform class name, plus any init kwargs.
    """

    def __init__(self, pipelines):
        assert isinstance(pipelines, list)
        self.transforms = []
        for cfg in pipelines:
            assert isinstance(cfg, dict) and "type" in cfg
            cfg = cfg.copy()
            cls_name = cfg.pop("type")
            if cls_name not in _PIPELINE_REGISTRY:
                raise KeyError(
                    f'Pipeline transform "{cls_name}" not found in registry. Available: {list(_PIPELINE_REGISTRY.keys())}'
                )
            self.transforms.append(_PIPELINE_REGISTRY[cls_name](**cfg))

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += f"\n    {t}"
        format_string += "\n)"
        return format_string


@register_pipeline
class CustomLoadMultiViewImageFromFiles:
    """Load multi-view images from a list of file paths.

    Expects results['img_filename'] to be a list of filenames.
    """

    def __init__(self, to_float32=False, color_type="unchanged"):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        filename = results["img_filename"]
        img = [imread(name, self.color_type) for name in filename]
        if self.to_float32:
            img = [_.astype(np.float32) for _ in img]
        results["filename"] = filename
        results["img"] = img
        results["img_shape"] = [img_.shape for img_ in img]
        results["ori_shape"] = [img_.shape for img_ in img]
        results["pad_shape"] = [img_.shape for img_ in img]
        results["crop_shape"] = [np.zeros(2) for _ in img]
        results["scale_factor"] = [1.0 for _ in img]
        num_channels = 1 if len(img[0].shape) < 3 else img[0].shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(to_float32={self.to_float32}, color_type='{self.color_type}')"


@register_pipeline
class PadMultiViewImageSame2Max:
    """Pad all multi-view images to the max height/width, then optionally to a divisor."""

    def __init__(self, size_divisor=None, pad_val=0):
        self.size_divisor = size_divisor
        self.pad_val = pad_val

    def _pad_img(self, results):
        max_h = max((img.shape[0] for img in results["img"]))
        max_w = max((img.shape[1] for img in results["img"]))
        padded_img = [
            impad(img, shape=(max_h, max_w), pad_val=self.pad_val) for img in results["img"]
        ]
        if self.size_divisor is not None:
            padded_img = [
                impad_to_multiple(img, self.size_divisor, pad_val=self.pad_val)
                for img in padded_img
            ]
        results["img"] = padded_img
        results["pad_shape"] = [img.shape for img in padded_img]
        results["pad_fixed_size"] = None
        results["pad_size_divisor"] = self.size_divisor

    def __call__(self, results):
        self._pad_img(results)
        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(size_divisor={self.size_divisor}, pad_val={self.pad_val})"
        )


@register_pipeline
class NormalizeMultiviewImage:
    """Normalize multi-view images with mean and std."""

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        results["img"] = [
            imnormalize(img, self.mean, self.std, self.to_rgb) for img in results["img"]
        ]
        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"


@register_pipeline
class RandomScaleImageMultiViewImage:
    """Randomly scale multi-view images and adjust intrinsics accordingly."""

    def __init__(self, scales=[]):
        self.scales = scales
        assert len(self.scales) == 1

    def _resize_bboxes(self, results, scale_factor):
        if "gt_bboxes" in results:
            results["gt_bboxes"] = results["gt_bboxes"] * scale_factor

    def __call__(self, results):
        rand_scale = np.random.choice(self.scales)
        y_size = [int(img.shape[0] * rand_scale) for img in results["img"]]
        x_size = [int(img.shape[1] * rand_scale) for img in results["img"]]
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= rand_scale
        scale_factor[1, 1] *= rand_scale
        results["img"] = [
            imresize(img, (x_size[i], y_size[i]), return_scale=False)
            for i, img in enumerate(results["img"])
        ]
        lidar2img = [scale_factor @ l2i for l2i in results["lidar2img"]]
        cam_intrinsic = [scale_factor @ c2i for c2i in results["cam_intrinsic"]]
        results["lidar2img"] = lidar2img
        results["cam_intrinsic"] = cam_intrinsic
        results["img_shape"] = [img.shape for img in results["img"]]
        results["ori_shape"] = [img.shape for img in results["img"]]
        results["scale_factor"] = [rand_scale for _ in results["img"]]
        self._resize_bboxes(results, rand_scale)
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(scales={self.scales})"


@register_pipeline
class CropFrontViewImageForAv2:
    """Crop the front center view image for Argoverse2 dataset."""

    def __init__(self, crop_h=(356, 1906)):
        self.crop_h = crop_h

    def _crop_img(self, results):
        results["ori_shape"] = [img.shape for img in results["img"]]
        results["img"][0] = results["img"][0][self.crop_h[0] : self.crop_h[1]]
        results["img_shape"] = [img.shape for img in results["img"]]
        results["crop_shape"][0] = np.array([0, self.crop_h[0]])

    def _crop_cam_intrinsic(self, results):
        results["cam_intrinsic"][0][1, 2] -= self.crop_h[0]
        results["lidar2img"][0] = results["cam_intrinsic"][0] @ results["lidar2cam"][0]

    def _crop_bbox(self, results):
        if "gt_bboxes" in results:
            results["gt_bboxes"][:, 1] -= self.crop_h[0]
            results["gt_bboxes"][:, 3] -= self.crop_h[0]
            mask = results["gt_bboxes"][:, 3] > 0
            results["gt_bboxes"] = results["gt_bboxes"][mask]
            results["gt_labels"] = results["gt_labels"][mask]
            if "gt_lane_lste_adj" in results:
                results["gt_lane_lste_adj"] = results["gt_lane_lste_adj"][:, mask]

    def __call__(self, results):
        self._crop_img(results)
        self._crop_cam_intrinsic(results)
        self._crop_bbox(results)
        return results


@register_pipeline
class CustomFormatBundle3DLane:
    """Format data for the 3D Lane SegNet model.

    Converts numpy arrays to tensors and wraps them.  In this remake
    version DataContainer is replaced by plain dicts / tensors.
    """

    def __init__(self, class_names, with_gt=True, with_label=True):
        self.class_names = class_names
        self.with_gt = with_gt
        self.with_label = with_label

    def __call__(self, results):
        images = [image.transpose(2, 0, 1) for image in results["img"]]
        results["img"] = torch.from_numpy(np.ascontiguousarray(np.stack(images)))
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(class_names={self.class_names}, with_gt={self.with_gt}, with_label={self.with_label})"


@register_pipeline
class CustomCollect3D:
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline.
    Collects specified keys and meta_keys into the output dict.
    In this remake version, no DataContainer wrapping is performed;
    img_metas is stored as a plain dict.
    """

    def __init__(
        self,
        keys,
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "lidar2img",
            "depth2img",
            "cam2img",
            "pad_shape",
            "crop_shape",
            "scale_factor",
            "flip",
            "pcd_horizontal_flip",
            "pcd_vertical_flip",
            "box_mode_3d",
            "box_type_3d",
            "img_norm_cfg",
            "pcd_trans",
            "sample_idx",
            "prev_idx",
            "next_idx",
            "pcd_scale_factor",
            "pcd_rotation",
            "pts_filename",
            "transformation_3d_flow",
            "scene_token",
            "can_bus",
            "lidar2global_rotation",
            "lidar2global_translation",
            "prev",
            "next",
            "lane_id",
        ),
    ):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        img_metas = {}
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]
        data["img_metas"] = img_metas
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(keys={self.keys}, meta_keys={self.meta_keys})"


class OpenLaneV2SubsetALaneSegNetDataset(Dataset):
    """Sequential OpenLane-V2 SubsetA inference dataset using LaneSegNet metadata. No GT targets or future poses are consumed."""

    CAMS = (
        "ring_front_center",
        "ring_front_left",
        "ring_front_right",
        "ring_rear_left",
        "ring_rear_right",
        "ring_side_left",
        "ring_side_right",
    )
    LANE_CLASSES = ("lane_segment", "ped_crossing")
    TE_CLASSES = ("traffic_light", "road_sign")
    TE_ATTR_CLASSES = (
        "unknown",
        "red",
        "green",
        "yellow",
        "go_straight",
        "turn_left",
        "turn_right",
        "no_left_turn",
        "no_right_turn",
        "u_turn",
        "no_u_turn",
        "slight_left",
        "slight_right",
    )
    MAP_CHANGE_LOGS = [
        "75e8adad-50a6-3245-8726-5e612db3d165",
        "54bc6dbc-ebfb-3fba-b5b3-57f88b4b79ca",
        "af170aac-8465-3d7b-82c5-64147e94af7d",
        "6e106cf8-f6dd-38f6-89c8-9be7a71e7275",
    ]

    def __init__(
        self, data_root, ann_file, pipeline, classes=None, modality=None, points_num=10, split="val"
    ):
        super().__init__()
        self.data_root = str(data_root)
        self.ann_file = str(ann_file)
        self.test_mode = True
        self.modality = modality or dict(use_camera=True)
        self.split = split
        self.points_num = points_num
        self.CLASSES = self.get_classes(classes)
        self.LANE_CLASSES = self.CLASSES
        self.data_infos = self.load_annotations(self.ann_file)
        seen = set()
        previous_scene = None
        previous_timestamp = None
        for index, info in enumerate(self.data_infos):
            scene = info["segment_id"]
            timestamp = int(info["timestamp"])
            if scene != previous_scene:
                if scene in seen:
                    raise ValueError("Scenes must be contiguous, not interleaved")
                seen.add(scene)
                previous_timestamp = None
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise ValueError("Frames must be in ascending timestamp order within each scene")
            info["sample_idx"] = index
            previous_scene, previous_timestamp = (scene, timestamp)
        self.pipeline = Compose(pipeline)
        self._set_sequence_info()

    def _set_sequence_info(self):
        """Compute idx2scene mapping for temporal propagation."""
        scene_name2idx = {}
        for idx in range(len(self.data_infos)):
            scene = self.data_infos[idx]["segment_id"]
            if scene not in scene_name2idx:
                scene_name2idx[scene] = []
            scene_name2idx[scene].append(idx)
        self.scene_name2idx = scene_name2idx
        idx2scene = {}
        for scene_name, scene_info in scene_name2idx.items():
            for local_idx, global_idx in enumerate(scene_info):
                idx2scene[global_idx] = (scene_name, local_idx, len(scene_info))
        self.idx2scene = idx2scene

    def load_annotations(self, ann_file):
        with open(ann_file, "rb") as handle:
            infos = pickle.load(handle)
        return list(infos.values()) if isinstance(infos, dict) else infos

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information for the pipeline, or None if filtered.
        """
        info = self.data_infos[index]
        input_dict = dict(sample_idx=info["sample_idx"], scene_token=info["segment_id"])
        if self.modality.get("use_camera", False):
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_name, cam_info in info["sensor"].items():
                image_path = cam_info["image_path"]
                image_paths.append(os.path.join(self.data_root, image_path))
                lidar2cam_r = np.linalg.inv(cam_info["extrinsic"]["rotation"])
                lidar2cam_t = cam_info["extrinsic"]["translation"] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = np.array(cam_info["intrinsic"]["K"])
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                lidar2img_rt = viewpad @ lidar2cam_rt.T
                lidar2img_rts.append(lidar2img_rt)
                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)
            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                )
            )
        can_bus = np.zeros(18)
        rotation = Quaternion._from_matrix(np.array(info["pose"]["rotation"]))
        can_bus[:3] = info["pose"]["translation"]
        can_bus[3:7] = rotation
        patch_angle = rotation.yaw_pitch_roll[0] / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle
        input_dict["can_bus"] = can_bus
        input_dict["lidar2global_rotation"] = np.array(info["pose"]["rotation"])
        input_dict["lidar2global_translation"] = np.array(info["pose"]["translation"])
        return input_dict

    def pre_pipeline(self, results):
        """Initialize result dict fields before pipeline execution."""
        results["img_fields"] = []
        results["bbox3d_fields"] = []
        results["pts_mask_fields"] = []
        results["pts_seg_fields"] = []
        results["bbox_fields"] = []
        results["mask_fields"] = []
        results["seg_fields"] = []

    def prepare_test_data(self, index):
        """Prepare a single test sample.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Test data dict.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        seq_info = self.idx2scene[index]
        example["seq_info"] = seq_info
        return example

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): Class names override.

        Returns:
            list[str]: A list of class names.
        """
        if classes is None:
            return cls.LANE_CLASSES
        if isinstance(classes, str):
            with open(classes, "r") as f:
                class_names = [line.strip() for line in f.readlines() if line.strip()]
            return class_names
        elif isinstance(classes, (tuple, list)):
            return list(classes)
        else:
            raise ValueError(f"Unsupported type {type(classes)} of classes.")

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        return self.prepare_test_data(idx)

    def format_results(self, results, jsonfile_prefix=None):
        """Format model predictions into the OpenLane V2 evaluation format.

        Args:
            results (list[dict]): Model output results per sample.
            jsonfile_prefix (str | None): Not used; kept for API compatibility.

        Returns:
            dict: Formatted prediction dict.
        """
        pred_dict = {}
        pred_dict["method"] = "DualTopo"
        pred_dict["authors"] = []
        pred_dict["e-mail"] = ""
        pred_dict["institution / company"] = ""
        pred_dict["country / region"] = "CN"
        pred_dict["results"] = {}
        for idx, result in enumerate(results):
            info = self.data_infos[idx]
            key = (self.split, info["segment_id"], str(info["timestamp"]))
            pred_info = dict(
                lane_segment=[], area=[], traffic_element=[], topology_lsls=None, topology_lste=None
            )
            if result["lane_results"] is not None:
                lane_results = result["lane_results"]
                scores = lane_results[1]
                valid_indices = np.argsort(-scores)
                lanes = lane_results[0][valid_indices]
                labels = lane_results[2][valid_indices]
                scores = scores[valid_indices]
                lanes = lanes.reshape(-1, lanes.shape[-1] // 3, 3)
                lane_results[3][valid_indices]
                left_type_labels = lane_results[4][valid_indices]
                lane_results[5][valid_indices]
                right_type_labels = lane_results[6][valid_indices]
                pred_area_index = []
                for pred_idx, (lane, score, label) in enumerate(zip(lanes, scores, labels)):
                    if label == 0:
                        points = lane.astype(np.float32)
                        pred_lane_segment = {}
                        pred_lane_segment["id"] = 20000 + pred_idx
                        pred_lane_segment["centerline"] = fix_pts_interpolate(
                            points[: self.points_num], 10
                        )
                        pred_lane_segment["left_laneline"] = fix_pts_interpolate(
                            points[self.points_num : self.points_num * 2], 10
                        )
                        pred_lane_segment["right_laneline"] = fix_pts_interpolate(
                            points[self.points_num * 2 :], 10
                        )
                        pred_lane_segment["left_laneline_type"] = left_type_labels[pred_idx]
                        pred_lane_segment["right_laneline_type"] = right_type_labels[pred_idx]
                        pred_lane_segment["confidence"] = score.item()
                        pred_info["lane_segment"].append(pred_lane_segment)
                    elif label == 1:
                        points = lane.astype(np.float32)
                        pred_ped = {}
                        pred_ped["id"] = 20000 + pred_idx
                        pred_points = np.concatenate(
                            (
                                fix_pts_interpolate(
                                    points[self.points_num : self.points_num * 2], 10
                                ),
                                fix_pts_interpolate(points[self.points_num * 2 :][::-1], 10),
                            ),
                            axis=0,
                        )
                        pred_ped["points"] = pred_points
                        pred_ped["category"] = label
                        pred_ped["confidence"] = score.item()
                        pred_info["area"].append(pred_ped)
                        pred_area_index.append(pred_idx)
                    elif label == 2:
                        raise NotImplementedError
            if result.get("bbox_results") is not None:
                te_results = result["bbox_results"]
                scores = te_results[1]
                te_valid_indices = np.argsort(-scores)
                tes = te_results[0][te_valid_indices]
                scores = scores[te_valid_indices]
                class_idxs = te_results[2][te_valid_indices]
                for pred_idx, (te, score, class_idx) in enumerate(zip(tes, scores, class_idxs)):
                    te_info = dict(
                        id=20000 + pred_idx,
                        category=1 if class_idx < 4 else 2,
                        attribute=class_idx,
                        points=te.reshape(2, 2).astype(np.float32),
                        confidence=score,
                    )
                    pred_info["traffic_element"].append(te_info)
            valid_indices_local = None
            if result["lane_results"] is not None:
                lane_results = result["lane_results"]
                scores = lane_results[1]
                valid_indices_local = np.argsort(-scores)
            if result.get("lsls_results") is not None and valid_indices_local is not None:
                topology_lsls_area = result["lsls_results"].astype(np.float32)[valid_indices_local][
                    :, valid_indices_local
                ]
                topology_lsls_area = np.delete(topology_lsls_area, pred_area_index, axis=0)
                topology_lsls = np.delete(topology_lsls_area, pred_area_index, axis=1)
                pred_info["topology_lsls"] = topology_lsls
            else:
                pred_info["topology_lsls"] = np.zeros(
                    (len(pred_info["lane_segment"]), len(pred_info["lane_segment"])),
                    dtype=np.float32,
                )
            if result.get("lste_results") is not None and valid_indices_local is not None:
                topology_lste_area = result["lste_results"].astype(np.float32)[valid_indices_local]
                topology_lste = np.delete(topology_lste_area, pred_area_index, axis=0)
                pred_info["topology_lste"] = topology_lste
            else:
                pred_info["topology_lste"] = np.zeros(
                    (len(pred_info["lane_segment"]), len(pred_info["traffic_element"])),
                    dtype=np.float32,
                )
            pred_dict["results"][key] = {"predictions": pred_info}
        return pred_dict


def collate_fn(batch):
    if len(batch) != 1:
        raise ValueError("Temporal inference requires sequential batch size 1")
    sample = batch[0]
    return dict(
        img=sample["img"].unsqueeze(0),
        img_metas=[sample["img_metas"]],
        seq_info=[sample["seq_info"]],
    )
