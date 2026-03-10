import inference as inference_pkg


def test_inference_public_api_exports_core_symbols() -> None:
    required = [
        "get_point_cloud",
        "get_input_data_from_point_cloud",
        "remove_plane",
        "get_rot_from_extrinsic",
        "unproject_kinect_depth",
    ]
    for name in required:
        assert hasattr(inference_pkg, name), f"Missing inference public symbol: {name}"
