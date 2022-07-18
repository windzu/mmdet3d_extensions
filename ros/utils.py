def convert_SyncBN(config):
    """Convert config's naiveSyncBN to BN.

    Args:
         config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
    """
    if isinstance(config, dict):
        for item in config:
            if item == "norm_cfg":
                config[item]["type"] = config[item]["type"].replace("naiveSyncBN", "BN")
            else:
                convert_SyncBN(config[item])


def convert_test_pipeline_load_file_to_pointcloud2(config):
    """Convert config's load file ways to pointcloud2.

    Args:
         config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
    """
    # # change test_pipeline's load file ways to pointcloud2
    # if isinstance(config, list):
    #     for item in config:
    #         if isinstance(item, dict):
    #             for key, value in item.items():
    #                 if key == "type" and value == "LoadPointsFromFile":
    #                     item[key] = "LoadPointsFromPointCloud2"

    for item in config.data.test.pipeline:
        if isinstance(item, dict):
            for key, value in item.items():
                if key == "type" and value == "LoadPointsFromFile":
                    item[key] = "LoadPointsFromPointCloud2"
