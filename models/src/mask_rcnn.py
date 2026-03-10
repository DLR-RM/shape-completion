from typing import Any

import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torchmetrics.detection import MeanAveragePrecision

from .model import Model


def load_mask_rcnn_model(from_pretrained: bool = True):
    from detectron2 import model_zoo
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model

    cfg = get_cfg()
    config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_file))

    # Set the threshold for detection
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # Load pre-trained weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)

    # Build the entire model
    model = build_model(cfg)

    if from_pretrained:
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

    return model, cfg


class MaskRCNN(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from detectron2.data import MetadataCatalog

        self.model, self.cfg = load_mask_rcnn_model()
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        self.map = MeanAveragePrecision(iou_type="segm", backend="faster_coco_eval")

    def forward(self, inputs: Tensor, **kwargs) -> list[dict[str, Any]]:
        model_inputs = [
            {
                "image": torch.flip(image, dims=[0]),  # Convert RGB to BGR
                "height": kwargs.get("height", image.shape[1]),
                "width": kwargs.get("width", image.shape[2]),
            }
            for image in inputs
        ]
        return self.model(model_inputs)

    def predict(self, data: dict[str, Tensor], **kwargs) -> list[dict[str, Any]]:
        return self(**data, **kwargs)

    def evaluate(self, data: dict[str, Tensor], show: bool = False, **kwargs) -> dict[str, Tensor]:
        instances = self.predict(data, **kwargs)

        preds_for_map = []
        targets_for_map = []
        for i, instance in enumerate(instances):
            instance = instance["instances"]

            preds_for_map.append(
                dict(
                    boxes=instance.pred_boxes.tensor,
                    masks=instance.pred_masks,
                    scores=instance.scores,
                    # labels=instance.pred_classes,
                    labels=torch.zeros_like(instance.scores, dtype=torch.long),
                )
            )

            targets_for_map.append(
                dict(
                    boxes=data["inputs.boxes"][i],
                    masks=data["inputs.masks"][i],
                    # labels=data["inputs.labels"][i],
                    labels=torch.zeros_like(data["inputs.labels"][i]),
                )
            )

            if show:
                from detectron2.utils.visualizer import Visualizer

                v = Visualizer(data["inputs"][i].permute(1, 2, 0).cpu().numpy(), self.metadata)
                out = v.draw_instance_predictions(instance.to("cpu"))
                plt.figure(figsize=(8, 8))
                plt.imshow(out.get_image())
                plt.axis("off")
                plt.tight_layout()
                plt.show()
        self.map.update(preds_for_map, targets_for_map)
        return {}

    def on_validation_epoch_end(self, *args, **kwargs) -> dict[str, Tensor]:
        map_results = self.map.compute()
        self.map.reset()
        return map_results

    def loss(self, *args, **kwargs):
        raise NotImplementedError
