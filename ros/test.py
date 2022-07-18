from argparse import ArgumentParser
import numpy as np
import torch
from copy import deepcopy

# mmlab
import mmcv
from mmcv.parallel import collate, scatter
from mmdet3d.apis import inference_detector, init_model, show_result_meshlab
from mmdet3d.core.points import BasePoints, get_points_type
from mmdet3d.datasets.builder import PIPELINES
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.core.bbox import get_box_type

# local
from utils import convert_SyncBN, convert_test_pipeline_load_file_to_pointcloud2

# ros
from pypcd import pypcd
from autoware_msgs.msg import DetectedObject, DetectedObjectArray
import rospy
from sensor_msgs.msg import PointCloud2

# register "LoadPointsFromPointCloud2" to the global registry
@PIPELINES.register_module()
class LoadPointsFromPointCloud2(object):
    """Load Points From LoadPointsFromPointCloud2.
    加载式 point cloud 2 格式的点云数据

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(
        self,
        coord_type,
        load_dim=6,
        use_dim=[0, 1, 2],
        shift_height=False,
        use_color=False,
        file_client_args=dict(backend="disk"),
    ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _pc_format_converter(self, input_data):
        """将输入的点云数据转换为符合mmdetection3d中模型输入需要的点云格式。
        并且点云目标检测模型因为训练时候所采用的数据集的不同，对输入点云的维度需求不同。
        因为实例化模型是根据配置文件来的，所以转换后的点云要与配置文件中一致，否则会导致维度出错。
        例如：
            nuscenes数据集的原始点云是[x, y, z, intensity, ring]5维,
        那么采用nuscenes数据集的配置文件所实例化的模型就需要提供5维的点云(尽管最后只是使用了其中的前4维参与了训练)
            waymo数据集的原始点云是[x, y, z, intensity,ring,未知]6维,
        那么采用waymo数据集的配置文件所实例化的模型就需要提供6维的点云(尽管最后只是使用了其中的前4维参与了训练)

        Args:
            input_data (PointCloud2): 输入的点云数据,来自ros1订阅,为ros1 sensor_msgs 的PointCloud2类型的点云数据
        """
        pc = pypcd.PointCloud.from_msg(input_data)
        # 所有点云至少包含xyz这三个维度
        x = pc.pc_data["x"].flatten()
        y = pc.pc_data["y"].flatten()
        z = pc.pc_data["z"].flatten()
        intensity = pc.pc_data["intensity"].flatten()
        ring = None

        # 如果点云数据中包含了ring维度，则将其加入到输出点云中
        if "ring" in pc.pc_data:
            ring = pc.pc_data["ring"].flatten()

        # 首先将输入的点云数据转换为6维的点云，原始点云中不存在的维度设置为0，然后根据需求转换为需要的维度
        pc_array_6d = np.zeros((x.shape[0], 6))
        pc_array_6d[:, 0] = x
        pc_array_6d[:, 1] = y
        pc_array_6d[:, 2] = z
        pc_array_6d[:, 3] = intensity
        if ring is not None:
            pc_array_6d[:, 4] = ring

        # 接下来根据需求转换为需要的维度
        # 注意！！，模型所需的输入是1维的点云，所以记得要将维度转换为1维

        return pc_array_6d

    def _load_pointcloud2(self, pointcloud2):
        """Private function to load point clouds data.

        Args:
            pointcloud2 (pointcloud2): pointcloud2 format point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """

        points = self._pc_format_converter(pointcloud2)

        return points

    def __call__(self, results):
        """Call function to load points data from pointcloud2.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pointcloud2 = results["pointcloud2"]
        points = self._load_pointcloud2(pointcloud2)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate([points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]
                )
            )

        points_class = get_points_type(self.coord_type)
        points = points_class(points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results["points"] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + "("
        repr_str += f"shift_height={self.shift_height}, "
        repr_str += f"use_color={self.use_color}, "
        repr_str += f"file_client_args={self.file_client_args}, "
        repr_str += f"load_dim={self.load_dim}, "
        repr_str += f"use_dim={self.use_dim})"
        return repr_str


class ROSExtension:
    """ROS的一个扩展,具体内容还未确定,首先先基于mmdet3d完成对点云处理的拓展"""

    def __init__(
        self,
        model,
        lidar_topic,
        detected_objects_topic,
    ):
        self.model = model
        self.lidar_topic = lidar_topic
        self.detected_objects_topic = detected_objects_topic

        self.device = next(self.model.parameters()).device
        self.cfg = self.model.cfg
        self.cfg = self.cfg.copy()
        ## build the data pipeline
        self.test_pipeline = deepcopy(self.cfg.data.test.pipeline)
        self.test_pipeline = Compose(self.test_pipeline)

        # debug
        print("test_pipeline:", self.test_pipeline)
        self.box_type_3d, self.box_mode_3d = get_box_type(self.cfg.data.test.box_type_3d)

    def start(self):
        rospy.init_node("detection", anonymous=True)
        self.detected_objects_publisher = rospy.Publisher(
            self.detected_objects_topic, DetectedObjectArray, queue_size=1
        )
        self.lidar_subscriber = rospy.Subscriber(self.lidar_topic, PointCloud2, self.__callback)
        rospy.spin()

    def __callback(self, pointcloud2):
        # self.data["pointcloud2"] = pointcloud2
        self.data = self.__create_data(pointcloud2)
        self.data = self.test_pipeline(self.data)
        self.data = collate([self.data], samples_per_gpu=1)

        if next(self.model.parameters()).is_cuda:
            # scatter to specified GPU
            self.data = scatter(self.data, [self.device.index])[0]
        else:
            # this is a workaround to avoid the bug of MMDataParallel
            self.data["img_metas"] = self.data["img_metas"][0].data
            self.data["points"] = self.data["points"][0].data

        with torch.no_grad():
            result = self.model(return_loss=False, rescale=True, **self.data)

        # debug
        print(result)

    def __create_data(self, pointcloud2):
        data = dict(
            pointcloud2=pointcloud2,
            box_type_3d=self.box_type_3d,
            box_mode_3d=self.box_mode_3d,
            # for ScanNet demo we need axis_align_matrix
            ann_info=dict(axis_align_matrix=np.eye(4)),
            sweeps=[],
            # set timestamp = 0
            timestamp=[0],
            img_fields=[],
            bbox3d_fields=[],
            pts_mask_fields=[],
            pts_seg_fields=[],
            bbox_fields=[],
            mask_fields=[],
            seg_fields=[],
        )
        return data


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("config", help="Config file")
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument("--score-thr", type=float, default=0.0, help="bbox score threshold")
    parser.add_argument("--out-dir", type=str, default="demo", help="dir to save results")
    parser.add_argument("--show", action="store_true", help="show online visualization results")
    parser.add_argument("--snapshot", action="store_true", help="whether to save online visualization results")
    args = parser.parse_args()
    return args


def main():
    # args = parse_args()

    # just for test
    # load config
    config_path = "../mmdetection3d/configs/pointpillars/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d.py"
    checkpoint_path = "../mmdetection3d/checkpoints/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth"
    device = "cuda:0"

    # change config's load file to pointcloud2
    config = mmcv.Config.fromfile(config_path)
    convert_test_pipeline_load_file_to_pointcloud2(config)
    print(config.data.test.pipeline)

    # init model
    model = init_model(config=config, checkpoint=checkpoint_path, device=device)

    # debug
    lidar_frame_id = "LIDAR_TOP"
    lidar_topic = "/" + lidar_frame_id
    lidar_detected_objects_topic = lidar_topic + "/detected_objects"
    ros_extension = ROSExtension(
        model=model, detected_objects_topic=lidar_detected_objects_topic, lidar_topic=lidar_topic
    )
    ros_extension.start()


if __name__ == "__main__":
    main()
