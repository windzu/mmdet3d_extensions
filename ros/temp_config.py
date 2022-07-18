{
    "voxel_size": [0.25, 0.25, 8],
    "model": {
        "type": "MVXFasterRCNN",
        "pts_voxel_layer": {
            "max_num_points": 64,
            "point_cloud_range": [-50, -50, -5, 50, 50, 3],
            "voxel_size": [0.25, 0.25, 8],
            "max_voxels": (30000, 40000),
        },
        "pts_voxel_encoder": {
            "type": "HardVFE",
            "in_channels": 4,
            "feat_channels": [64, 64],
            "with_distance": False,
            "voxel_size": [0.25, 0.25, 8],
            "with_cluster_center": True,
            "with_voxel_center": True,
            "point_cloud_range": [-50, -50, -5, 50, 50, 3],
            "norm_cfg": {"type": "naiveSyncBN1d", "eps": 0.001, "momentum": 0.01},
        },
        "pts_middle_encoder": {"type": "PointPillarsScatter", "in_channels": 64, "output_shape": [400, 400]},
        "pts_backbone": {
            "type": "SECOND",
            "in_channels": 64,
            "norm_cfg": {"type": "naiveSyncBN2d", "eps": 0.001, "momentum": 0.01},
            "layer_nums": [3, 5, 5],
            "layer_strides": [2, 2, 2],
            "out_channels": [64, 128, 256],
        },
        "pts_neck": {
            "type": "FPN",
            "norm_cfg": {"type": "naiveSyncBN2d", "eps": 0.001, "momentum": 0.01},
            "act_cfg": {"type": "ReLU"},
            "in_channels": [64, 128, 256],
            "out_channels": 256,
            "start_level": 0,
            "num_outs": 3,
        },
        "pts_bbox_head": {
            "type": "Anchor3DHead",
            "num_classes": 10,
            "in_channels": 256,
            "feat_channels": 256,
            "use_direction_classifier": True,
            "anchor_generator": {
                "type": "AlignedAnchor3DRangeGenerator",
                "ranges": [[-50, -50, -1.8, 50, 50, -1.8]],
                "scales": [1, 2, 4],
                "sizes": [[2.5981, 0.866, 1.0], [1.7321, 0.5774, 1.0], [1.0, 1.0, 1.0], [0.4, 0.4, 1]],
                "custom_values": [0, 0],
                "rotations": [0, 1.57],
                "reshape_out": True,
            },
            "assigner_per_size": False,
            "diff_rad_by_sin": True,
            "dir_offset": -0.7854,
            "bbox_coder": {"type": "DeltaXYZWLHRBBoxCoder", "code_size": 9},
            "loss_cls": {"type": "FocalLoss", "use_sigmoid": True, "gamma": 2.0, "alpha": 0.25, "loss_weight": 1.0},
            "loss_bbox": {"type": "SmoothL1Loss", "beta": 0.1111111111111111, "loss_weight": 1.0},
            "loss_dir": {"type": "CrossEntropyLoss", "use_sigmoid": False, "loss_weight": 0.2},
        },
        "train_cfg": {
            "pts": {
                "assigner": {
                    "type": "MaxIoUAssigner",
                    "iou_calculator": {"type": "BboxOverlapsNearest3D"},
                    "pos_iou_thr": 0.6,
                    "neg_iou_thr": 0.3,
                    "min_pos_iou": 0.3,
                    "ignore_iof_thr": -1,
                },
                "allowed_border": 0,
                "code_weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
                "pos_weight": -1,
                "debug": False,
            }
        },
        "test_cfg": {
            "pts": {
                "use_rotate_nms": True,
                "nms_across_levels": False,
                "nms_pre": 1000,
                "nms_thr": 0.2,
                "score_thr": 0.05,
                "min_bbox_size": 0,
                "max_num": 500,
            }
        },
    },
    "point_cloud_range": [-50, -50, -5, 50, 50, 3],
    "class_names": [
        "car",
        "truck",
        "trailer",
        "bus",
        "construction_vehicle",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "barrier",
    ],
    "dataset_type": "NuScenesDataset",
    "data_root": "data/nuscenes/",
    "input_modality": {
        "use_lidar": True,
        "use_camera": False,
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    },
    "file_client_args": {"backend": "disk"},
    "train_pipeline": [
        {
            "type": "LoadPointsFromFile",
            "coord_type": "LIDAR",
            "load_dim": 5,
            "use_dim": 5,
            "file_client_args": {"backend": "disk"},
        },
        {"type": "LoadPointsFromMultiSweeps", "sweeps_num": 10, "file_client_args": {"backend": "disk"}},
        {"type": "LoadAnnotations3D", "with_bbox_3d": True, "with_label_3d": True},
        {
            "type": "GlobalRotScaleTrans",
            "rot_range": [-0.3925, 0.3925],
            "scale_ratio_range": [0.95, 1.05],
            "translation_std": [0, 0, 0],
        },
        {"type": "RandomFlip3D", "flip_ratio_bev_horizontal": 0.5},
        {"type": "PointsRangeFilter", "point_cloud_range": [-50, -50, -5, 50, 50, 3]},
        {"type": "ObjectRangeFilter", "point_cloud_range": [-50, -50, -5, 50, 50, 3]},
        {
            "type": "ObjectNameFilter",
            "classes": [
                "car",
                "truck",
                "trailer",
                "bus",
                "construction_vehicle",
                "bicycle",
                "motorcycle",
                "pedestrian",
                "traffic_cone",
                "barrier",
            ],
        },
        {"type": "PointShuffle"},
        {
            "type": "DefaultFormatBundle3D",
            "class_names": [
                "car",
                "truck",
                "trailer",
                "bus",
                "construction_vehicle",
                "bicycle",
                "motorcycle",
                "pedestrian",
                "traffic_cone",
                "barrier",
            ],
        },
        {"type": "Collect3D", "keys": ["points", "gt_bboxes_3d", "gt_labels_3d"]},
    ],
    "test_pipeline": [
        {
            "type": "LoadPointsFromFile",
            "coord_type": "LIDAR",
            "load_dim": 5,
            "use_dim": 5,
            "file_client_args": {"backend": "disk"},
        },
        {"type": "LoadPointsFromMultiSweeps", "sweeps_num": 10, "file_client_args": {"backend": "disk"}},
        {
            "type": "MultiScaleFlipAug3D",
            "img_scale": (1333, 800),
            "pts_scale_ratio": 1,
            "flip": False,
            "transforms": [
                {
                    "type": "GlobalRotScaleTrans",
                    "rot_range": [0, 0],
                    "scale_ratio_range": [1.0, 1.0],
                    "translation_std": [0, 0, 0],
                },
                {"type": "RandomFlip3D"},
                {"type": "PointsRangeFilter", "point_cloud_range": [-50, -50, -5, 50, 50, 3]},
                {
                    "type": "DefaultFormatBundle3D",
                    "class_names": [
                        "car",
                        "truck",
                        "trailer",
                        "bus",
                        "construction_vehicle",
                        "bicycle",
                        "motorcycle",
                        "pedestrian",
                        "traffic_cone",
                        "barrier",
                    ],
                    "with_label": False,
                },
                {"type": "Collect3D", "keys": ["points"]},
            ],
        },
    ],
    "eval_pipeline": [
        {
            "type": "LoadPointsFromFile",
            "coord_type": "LIDAR",
            "load_dim": 5,
            "use_dim": 5,
            "file_client_args": {"backend": "disk"},
        },
        {"type": "LoadPointsFromMultiSweeps", "sweeps_num": 10, "file_client_args": {"backend": "disk"}},
        {
            "type": "DefaultFormatBundle3D",
            "class_names": [
                "car",
                "truck",
                "trailer",
                "bus",
                "construction_vehicle",
                "bicycle",
                "motorcycle",
                "pedestrian",
                "traffic_cone",
                "barrier",
            ],
            "with_label": False,
        },
        {"type": "Collect3D", "keys": ["points"]},
    ],
    "data": {
        "samples_per_gpu": 2,
        "workers_per_gpu": 2,
        "train": {
            "type": "NuScenesDataset",
            "data_root": "data/nuscenes/",
            "ann_file": "data/nuscenes/nuscenes_infos_train.pkl",
            "pipeline": [
                {
                    "type": "LoadPointsFromFile",
                    "coord_type": "LIDAR",
                    "load_dim": 5,
                    "use_dim": 5,
                    "file_client_args": {"backend": "disk"},
                },
                {"type": "LoadPointsFromMultiSweeps", "sweeps_num": 10, "file_client_args": {"backend": "disk"}},
                {"type": "LoadAnnotations3D", "with_bbox_3d": True, "with_label_3d": True},
                {
                    "type": "GlobalRotScaleTrans",
                    "rot_range": [-0.3925, 0.3925],
                    "scale_ratio_range": [0.95, 1.05],
                    "translation_std": [0, 0, 0],
                },
                {"type": "RandomFlip3D", "flip_ratio_bev_horizontal": 0.5},
                {"type": "PointsRangeFilter", "point_cloud_range": [-50, -50, -5, 50, 50, 3]},
                {"type": "ObjectRangeFilter", "point_cloud_range": [-50, -50, -5, 50, 50, 3]},
                {
                    "type": "ObjectNameFilter",
                    "classes": [
                        "car",
                        "truck",
                        "trailer",
                        "bus",
                        "construction_vehicle",
                        "bicycle",
                        "motorcycle",
                        "pedestrian",
                        "traffic_cone",
                        "barrier",
                    ],
                },
                {"type": "PointShuffle"},
                {
                    "type": "DefaultFormatBundle3D",
                    "class_names": [
                        "car",
                        "truck",
                        "trailer",
                        "bus",
                        "construction_vehicle",
                        "bicycle",
                        "motorcycle",
                        "pedestrian",
                        "traffic_cone",
                        "barrier",
                    ],
                },
                {"type": "Collect3D", "keys": ["points", "gt_bboxes_3d", "gt_labels_3d"]},
            ],
            "classes": [
                "car",
                "truck",
                "trailer",
                "bus",
                "construction_vehicle",
                "bicycle",
                "motorcycle",
                "pedestrian",
                "traffic_cone",
                "barrier",
            ],
            "modality": {
                "use_lidar": True,
                "use_camera": False,
                "use_radar": False,
                "use_map": False,
                "use_external": False,
            },
            "test_mode": False,
            "box_type_3d": "LiDAR",
        },
        "val": {
            "type": "NuScenesDataset",
            "data_root": "data/nuscenes/",
            "ann_file": "data/nuscenes/nuscenes_infos_val.pkl",
            "pipeline": [
                {
                    "type": "LoadPointsFromFile",
                    "coord_type": "LIDAR",
                    "load_dim": 5,
                    "use_dim": 5,
                    "file_client_args": {"backend": "disk"},
                },
                {"type": "LoadPointsFromMultiSweeps", "sweeps_num": 10, "file_client_args": {"backend": "disk"}},
                {
                    "type": "MultiScaleFlipAug3D",
                    "img_scale": (1333, 800),
                    "pts_scale_ratio": 1,
                    "flip": False,
                    "transforms": [
                        {
                            "type": "GlobalRotScaleTrans",
                            "rot_range": [0, 0],
                            "scale_ratio_range": [1.0, 1.0],
                            "translation_std": [0, 0, 0],
                        },
                        {"type": "RandomFlip3D"},
                        {"type": "PointsRangeFilter", "point_cloud_range": [-50, -50, -5, 50, 50, 3]},
                        {
                            "type": "DefaultFormatBundle3D",
                            "class_names": [
                                "car",
                                "truck",
                                "trailer",
                                "bus",
                                "construction_vehicle",
                                "bicycle",
                                "motorcycle",
                                "pedestrian",
                                "traffic_cone",
                                "barrier",
                            ],
                            "with_label": False,
                        },
                        {"type": "Collect3D", "keys": ["points"]},
                    ],
                },
            ],
            "classes": [
                "car",
                "truck",
                "trailer",
                "bus",
                "construction_vehicle",
                "bicycle",
                "motorcycle",
                "pedestrian",
                "traffic_cone",
                "barrier",
            ],
            "modality": {
                "use_lidar": True,
                "use_camera": False,
                "use_radar": False,
                "use_map": False,
                "use_external": False,
            },
            "test_mode": True,
            "box_type_3d": "LiDAR",
        },
        "test": {
            "type": "NuScenesDataset",
            "data_root": "data/nuscenes/",
            "ann_file": "data/nuscenes/nuscenes_infos_val.pkl",
            "pipeline": [
                {
                    "type": "LoadPointsFromFile",
                    "coord_type": "LIDAR",
                    "load_dim": 5,
                    "use_dim": 5,
                    "file_client_args": {"backend": "disk"},
                },
                {"type": "LoadPointsFromMultiSweeps", "sweeps_num": 10, "file_client_args": {"backend": "disk"}},
                {
                    "type": "MultiScaleFlipAug3D",
                    "img_scale": (1333, 800),
                    "pts_scale_ratio": 1,
                    "flip": False,
                    "transforms": [
                        {
                            "type": "GlobalRotScaleTrans",
                            "rot_range": [0, 0],
                            "scale_ratio_range": [1.0, 1.0],
                            "translation_std": [0, 0, 0],
                        },
                        {"type": "RandomFlip3D"},
                        {"type": "PointsRangeFilter", "point_cloud_range": [-50, -50, -5, 50, 50, 3]},
                        {
                            "type": "DefaultFormatBundle3D",
                            "class_names": [
                                "car",
                                "truck",
                                "trailer",
                                "bus",
                                "construction_vehicle",
                                "bicycle",
                                "motorcycle",
                                "pedestrian",
                                "traffic_cone",
                                "barrier",
                            ],
                            "with_label": False,
                        },
                        {"type": "Collect3D", "keys": ["points"]},
                    ],
                },
            ],
            "classes": [
                "car",
                "truck",
                "trailer",
                "bus",
                "construction_vehicle",
                "bicycle",
                "motorcycle",
                "pedestrian",
                "traffic_cone",
                "barrier",
            ],
            "modality": {
                "use_lidar": True,
                "use_camera": False,
                "use_radar": False,
                "use_map": False,
                "use_external": False,
            },
            "test_mode": True,
            "box_type_3d": "LiDAR",
        },
    },
    "evaluation": {
        "interval": 24,
        "pipeline": [
            {
                "type": "LoadPointsFromFile",
                "coord_type": "LIDAR",
                "load_dim": 5,
                "use_dim": 5,
                "file_client_args": {"backend": "disk"},
            },
            {"type": "LoadPointsFromMultiSweeps", "sweeps_num": 10, "file_client_args": {"backend": "disk"}},
            {
                "type": "DefaultFormatBundle3D",
                "class_names": [
                    "car",
                    "truck",
                    "trailer",
                    "bus",
                    "construction_vehicle",
                    "bicycle",
                    "motorcycle",
                    "pedestrian",
                    "traffic_cone",
                    "barrier",
                ],
                "with_label": False,
            },
            {"type": "Collect3D", "keys": ["points"]},
        ],
    },
    "optimizer": {"type": "AdamW", "lr": 0.001, "weight_decay": 0.01},
    "optimizer_config": {"grad_clip": {"max_norm": 35, "norm_type": 2}},
    "lr_config": {"policy": "step", "warmup": "linear", "warmup_iters": 1000, "warmup_ratio": 0.001, "step": [20, 23]},
    "momentum_config": None,
    "runner": {"type": "EpochBasedRunner", "max_epochs": 24},
    "checkpoint_config": {"interval": 1},
    "log_config": {"interval": 50, "hooks": [{"type": "TextLoggerHook"}, {"type": "TensorboardLoggerHook"}]},
    "dist_params": {"backend": "nccl"},
    "log_level": "INFO",
    "work_dir": None,
    "load_from": None,
    "resume_from": None,
    "workflow": [("train", 1)],
    "opencv_num_threads": 0,
    "mp_start_method": "fork",
    "fp16": {"loss_scale": 32.0},
}


{
    "voxel_size": [0.25, 0.25, 8],
    "model": {
        "type": "MVXFasterRCNN",
        "pts_voxel_layer": {
            "max_num_points": 64,
            "point_cloud_range": [-50, -50, -5, 50, 50, 3],
            "voxel_size": [0.25, 0.25, 8],
            "max_voxels": (30000, 40000),
        },
        "pts_voxel_encoder": {
            "type": "HardVFE",
            "in_channels": 4,
            "feat_channels": [64, 64],
            "with_distance": False,
            "voxel_size": [0.25, 0.25, 8],
            "with_cluster_center": True,
            "with_voxel_center": True,
            "point_cloud_range": [-50, -50, -5, 50, 50, 3],
            "norm_cfg": {"type": "BN1d", "eps": 0.001, "momentum": 0.01},
        },
        "pts_middle_encoder": {"type": "PointPillarsScatter", "in_channels": 64, "output_shape": [400, 400]},
        "pts_backbone": {
            "type": "SECOND",
            "in_channels": 64,
            "norm_cfg": {"type": "BN2d", "eps": 0.001, "momentum": 0.01},
            "layer_nums": [3, 5, 5],
            "layer_strides": [2, 2, 2],
            "out_channels": [64, 128, 256],
        },
        "pts_neck": {
            "type": "FPN",
            "norm_cfg": {"type": "BN2d", "eps": 0.001, "momentum": 0.01},
            "act_cfg": {"type": "ReLU"},
            "in_channels": [64, 128, 256],
            "out_channels": 256,
            "start_level": 0,
            "num_outs": 3,
        },
        "pts_bbox_head": {
            "type": "Anchor3DHead",
            "num_classes": 10,
            "in_channels": 256,
            "feat_channels": 256,
            "use_direction_classifier": True,
            "anchor_generator": {
                "type": "AlignedAnchor3DRangeGenerator",
                "ranges": [[-50, -50, -1.8, 50, 50, -1.8]],
                "scales": [1, 2, 4],
                "sizes": [[2.5981, 0.866, 1.0], [1.7321, 0.5774, 1.0], [1.0, 1.0, 1.0], [0.4, 0.4, 1]],
                "custom_values": [0, 0],
                "rotations": [0, 1.57],
                "reshape_out": True,
            },
            "assigner_per_size": False,
            "diff_rad_by_sin": True,
            "dir_offset": -0.7854,
            "bbox_coder": {"type": "DeltaXYZWLHRBBoxCoder", "code_size": 9},
            "loss_cls": {"type": "FocalLoss", "use_sigmoid": True, "gamma": 2.0, "alpha": 0.25, "loss_weight": 1.0},
            "loss_bbox": {"type": "SmoothL1Loss", "beta": 0.1111111111111111, "loss_weight": 1.0},
            "loss_dir": {"type": "CrossEntropyLoss", "use_sigmoid": False, "loss_weight": 0.2},
        },
        "train_cfg": None,
        "test_cfg": {
            "pts": {
                "use_rotate_nms": True,
                "nms_across_levels": False,
                "nms_pre": 1000,
                "nms_thr": 0.2,
                "score_thr": 0.05,
                "min_bbox_size": 0,
                "max_num": 500,
            }
        },
        "pretrained": None,
    },
    "point_cloud_range": [-50, -50, -5, 50, 50, 3],
    "class_names": [
        "car",
        "truck",
        "trailer",
        "bus",
        "construction_vehicle",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "barrier",
    ],
    "dataset_type": "NuScenesDataset",
    "data_root": "data/nuscenes/",
    "input_modality": {
        "use_lidar": True,
        "use_camera": False,
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    },
    "file_client_args": {"backend": "disk"},
    "train_pipeline": [
        {
            "type": "LoadPointsFromFile",
            "coord_type": "LIDAR",
            "load_dim": 5,
            "use_dim": 5,
            "file_client_args": {"backend": "disk"},
        },
        {"type": "LoadPointsFromMultiSweeps", "sweeps_num": 10, "file_client_args": {"backend": "disk"}},
        {"type": "LoadAnnotations3D", "with_bbox_3d": True, "with_label_3d": True},
        {
            "type": "GlobalRotScaleTrans",
            "rot_range": [-0.3925, 0.3925],
            "scale_ratio_range": [0.95, 1.05],
            "translation_std": [0, 0, 0],
        },
        {"type": "RandomFlip3D", "flip_ratio_bev_horizontal": 0.5},
        {"type": "PointsRangeFilter", "point_cloud_range": [-50, -50, -5, 50, 50, 3]},
        {"type": "ObjectRangeFilter", "point_cloud_range": [-50, -50, -5, 50, 50, 3]},
        {
            "type": "ObjectNameFilter",
            "classes": [
                "car",
                "truck",
                "trailer",
                "bus",
                "construction_vehicle",
                "bicycle",
                "motorcycle",
                "pedestrian",
                "traffic_cone",
                "barrier",
            ],
        },
        {"type": "PointShuffle"},
        {
            "type": "DefaultFormatBundle3D",
            "class_names": [
                "car",
                "truck",
                "trailer",
                "bus",
                "construction_vehicle",
                "bicycle",
                "motorcycle",
                "pedestrian",
                "traffic_cone",
                "barrier",
            ],
        },
        {"type": "Collect3D", "keys": ["points", "gt_bboxes_3d", "gt_labels_3d"]},
    ],
    "test_pipeline": [
        {
            "type": "LoadPointsFromPointCloud2",
            "coord_type": "LIDAR",
            "load_dim": 5,
            "use_dim": 5,
            "file_client_args": {"backend": "disk"},
        },
        {"type": "LoadPointsFromMultiSweeps", "sweeps_num": 10, "file_client_args": {"backend": "disk"}},
        {
            "type": "MultiScaleFlipAug3D",
            "img_scale": (1333, 800),
            "pts_scale_ratio": 1,
            "flip": False,
            "transforms": [
                {
                    "type": "GlobalRotScaleTrans",
                    "rot_range": [0, 0],
                    "scale_ratio_range": [1.0, 1.0],
                    "translation_std": [0, 0, 0],
                },
                {"type": "RandomFlip3D"},
                {"type": "PointsRangeFilter", "point_cloud_range": [-50, -50, -5, 50, 50, 3]},
                {
                    "type": "DefaultFormatBundle3D",
                    "class_names": [
                        "car",
                        "truck",
                        "trailer",
                        "bus",
                        "construction_vehicle",
                        "bicycle",
                        "motorcycle",
                        "pedestrian",
                        "traffic_cone",
                        "barrier",
                    ],
                    "with_label": False,
                },
                {"type": "Collect3D", "keys": ["points"]},
            ],
        },
    ],
    "eval_pipeline": [
        {
            "type": "LoadPointsFromFile",
            "coord_type": "LIDAR",
            "load_dim": 5,
            "use_dim": 5,
            "file_client_args": {"backend": "disk"},
        },
        {"type": "LoadPointsFromMultiSweeps", "sweeps_num": 10, "file_client_args": {"backend": "disk"}},
        {
            "type": "DefaultFormatBundle3D",
            "class_names": [
                "car",
                "truck",
                "trailer",
                "bus",
                "construction_vehicle",
                "bicycle",
                "motorcycle",
                "pedestrian",
                "traffic_cone",
                "barrier",
            ],
            "with_label": False,
        },
        {"type": "Collect3D", "keys": ["points"]},
    ],
    "data": {
        "samples_per_gpu": 2,
        "workers_per_gpu": 2,
        "train": {
            "type": "NuScenesDataset",
            "data_root": "data/nuscenes/",
            "ann_file": "data/nuscenes/nuscenes_infos_train.pkl",
            "pipeline": [
                {
                    "type": "LoadPointsFromFile",
                    "coord_type": "LIDAR",
                    "load_dim": 5,
                    "use_dim": 5,
                    "file_client_args": {"backend": "disk"},
                },
                {"type": "LoadPointsFromMultiSweeps", "sweeps_num": 10, "file_client_args": {"backend": "disk"}},
                {"type": "LoadAnnotations3D", "with_bbox_3d": True, "with_label_3d": True},
                {
                    "type": "GlobalRotScaleTrans",
                    "rot_range": [-0.3925, 0.3925],
                    "scale_ratio_range": [0.95, 1.05],
                    "translation_std": [0, 0, 0],
                },
                {"type": "RandomFlip3D", "flip_ratio_bev_horizontal": 0.5},
                {"type": "PointsRangeFilter", "point_cloud_range": [-50, -50, -5, 50, 50, 3]},
                {"type": "ObjectRangeFilter", "point_cloud_range": [-50, -50, -5, 50, 50, 3]},
                {
                    "type": "ObjectNameFilter",
                    "classes": [
                        "car",
                        "truck",
                        "trailer",
                        "bus",
                        "construction_vehicle",
                        "bicycle",
                        "motorcycle",
                        "pedestrian",
                        "traffic_cone",
                        "barrier",
                    ],
                },
                {"type": "PointShuffle"},
                {
                    "type": "DefaultFormatBundle3D",
                    "class_names": [
                        "car",
                        "truck",
                        "trailer",
                        "bus",
                        "construction_vehicle",
                        "bicycle",
                        "motorcycle",
                        "pedestrian",
                        "traffic_cone",
                        "barrier",
                    ],
                },
                {"type": "Collect3D", "keys": ["points", "gt_bboxes_3d", "gt_labels_3d"]},
            ],
            "classes": [
                "car",
                "truck",
                "trailer",
                "bus",
                "construction_vehicle",
                "bicycle",
                "motorcycle",
                "pedestrian",
                "traffic_cone",
                "barrier",
            ],
            "modality": {
                "use_lidar": True,
                "use_camera": False,
                "use_radar": False,
                "use_map": False,
                "use_external": False,
            },
            "test_mode": False,
            "box_type_3d": "LiDAR",
        },
        "val": {
            "type": "NuScenesDataset",
            "data_root": "data/nuscenes/",
            "ann_file": "data/nuscenes/nuscenes_infos_val.pkl",
            "pipeline": [
                {
                    "type": "LoadPointsFromFile",
                    "coord_type": "LIDAR",
                    "load_dim": 5,
                    "use_dim": 5,
                    "file_client_args": {"backend": "disk"},
                },
                {"type": "LoadPointsFromMultiSweeps", "sweeps_num": 10, "file_client_args": {"backend": "disk"}},
                {
                    "type": "MultiScaleFlipAug3D",
                    "img_scale": (1333, 800),
                    "pts_scale_ratio": 1,
                    "flip": False,
                    "transforms": [
                        {
                            "type": "GlobalRotScaleTrans",
                            "rot_range": [0, 0],
                            "scale_ratio_range": [1.0, 1.0],
                            "translation_std": [0, 0, 0],
                        },
                        {"type": "RandomFlip3D"},
                        {"type": "PointsRangeFilter", "point_cloud_range": [-50, -50, -5, 50, 50, 3]},
                        {
                            "type": "DefaultFormatBundle3D",
                            "class_names": [
                                "car",
                                "truck",
                                "trailer",
                                "bus",
                                "construction_vehicle",
                                "bicycle",
                                "motorcycle",
                                "pedestrian",
                                "traffic_cone",
                                "barrier",
                            ],
                            "with_label": False,
                        },
                        {"type": "Collect3D", "keys": ["points"]},
                    ],
                },
            ],
            "classes": [
                "car",
                "truck",
                "trailer",
                "bus",
                "construction_vehicle",
                "bicycle",
                "motorcycle",
                "pedestrian",
                "traffic_cone",
                "barrier",
            ],
            "modality": {
                "use_lidar": True,
                "use_camera": False,
                "use_radar": False,
                "use_map": False,
                "use_external": False,
            },
            "test_mode": True,
            "box_type_3d": "LiDAR",
        },
        "test": {
            "type": "NuScenesDataset",
            "data_root": "data/nuscenes/",
            "ann_file": "data/nuscenes/nuscenes_infos_val.pkl",
            "pipeline": [
                {
                    "type": "LoadPointsFromFile",
                    "coord_type": "LIDAR",
                    "load_dim": 5,
                    "use_dim": 5,
                    "file_client_args": {"backend": "disk"},
                },
                {"type": "LoadPointsFromMultiSweeps", "sweeps_num": 10, "file_client_args": {"backend": "disk"}},
                {
                    "type": "MultiScaleFlipAug3D",
                    "img_scale": (1333, 800),
                    "pts_scale_ratio": 1,
                    "flip": False,
                    "transforms": [
                        {
                            "type": "GlobalRotScaleTrans",
                            "rot_range": [0, 0],
                            "scale_ratio_range": [1.0, 1.0],
                            "translation_std": [0, 0, 0],
                        },
                        {"type": "RandomFlip3D"},
                        {"type": "PointsRangeFilter", "point_cloud_range": [-50, -50, -5, 50, 50, 3]},
                        {
                            "type": "DefaultFormatBundle3D",
                            "class_names": [
                                "car",
                                "truck",
                                "trailer",
                                "bus",
                                "construction_vehicle",
                                "bicycle",
                                "motorcycle",
                                "pedestrian",
                                "traffic_cone",
                                "barrier",
                            ],
                            "with_label": False,
                        },
                        {"type": "Collect3D", "keys": ["points"]},
                    ],
                },
            ],
            "classes": [
                "car",
                "truck",
                "trailer",
                "bus",
                "construction_vehicle",
                "bicycle",
                "motorcycle",
                "pedestrian",
                "traffic_cone",
                "barrier",
            ],
            "modality": {
                "use_lidar": True,
                "use_camera": False,
                "use_radar": False,
                "use_map": False,
                "use_external": False,
            },
            "test_mode": True,
            "box_type_3d": "LiDAR",
        },
    },
    "evaluation": {
        "interval": 24,
        "pipeline": [
            {
                "type": "LoadPointsFromFile",
                "coord_type": "LIDAR",
                "load_dim": 5,
                "use_dim": 5,
                "file_client_args": {"backend": "disk"},
            },
            {"type": "LoadPointsFromMultiSweeps", "sweeps_num": 10, "file_client_args": {"backend": "disk"}},
            {
                "type": "DefaultFormatBundle3D",
                "class_names": [
                    "car",
                    "truck",
                    "trailer",
                    "bus",
                    "construction_vehicle",
                    "bicycle",
                    "motorcycle",
                    "pedestrian",
                    "traffic_cone",
                    "barrier",
                ],
                "with_label": False,
            },
            {"type": "Collect3D", "keys": ["points"]},
        ],
    },
    "optimizer": {"type": "AdamW", "lr": 0.001, "weight_decay": 0.01},
    "optimizer_config": {"grad_clip": {"max_norm": 35, "norm_type": 2}},
    "lr_config": {"policy": "step", "warmup": "linear", "warmup_iters": 1000, "warmup_ratio": 0.001, "step": [20, 23]},
    "momentum_config": None,
    "runner": {"type": "EpochBasedRunner", "max_epochs": 24},
    "checkpoint_config": {"interval": 1},
    "log_config": {"interval": 50, "hooks": [{"type": "TextLoggerHook"}, {"type": "TensorboardLoggerHook"}]},
    "dist_params": {"backend": "nccl"},
    "log_level": "INFO",
    "work_dir": None,
    "load_from": None,
    "resume_from": None,
    "workflow": [("train", 1)],
    "opencv_num_threads": 0,
    "mp_start_method": "fork",
    "fp16": {"loss_scale": 32.0},
}