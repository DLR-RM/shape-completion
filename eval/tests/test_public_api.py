import eval as eval_pkg


def test_eval_public_api_exports_core_symbols() -> None:
    required = [
        "eval_mesh_pcd",
        "eval_mesh",
        "eval_pointcloud",
        "eval_occupancy",
        "EMPTY_RESULTS_DICT",
    ]
    for name in required:
        assert hasattr(eval_pkg, name), f"Missing eval public symbol: {name}"
