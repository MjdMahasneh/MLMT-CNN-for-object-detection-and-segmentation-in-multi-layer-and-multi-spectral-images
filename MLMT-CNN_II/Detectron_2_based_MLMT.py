import torch
import torch.nn as nn
import torchvision.models as models
import detectron2

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.structures import ImageList, Instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data.transforms import Resize, RandomFlip
from detectron2.modeling.backbone import Backbone
from detectron2.modeling import (
    META_ARCH_REGISTRY,
    build_backbone,
    build_proposal_generator,
    build_roi_heads,
    detector_postprocess
)

# Define a custom backbone that only returns the feature maps from a single intermediate stage of a standard backbone
class BandSpecificBackbone(Backbone):
    def __init__(self, cfg, input_shape, backbone_name):
        super().__init__()
        self.backbone = build_backbone(cfg, input_shape, backbone_name=backbone_name)
        output_shape = self.backbone.output_shape()
        self.output_shape = {k: output_shape[k] for k in cfg.MODEL.BACKBONE.OUT_FEATURES}

    def forward(self, images):
        features = self.backbone(images.tensor)
        return [features[k] for k in self.output_shape.keys()]


# Define the multi-band detection model
@META_ARCH_REGISTRY.register()
class MultiBandRCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Define the two band-specific backbones
        self.backbone1 = BandSpecificBackbone(cfg, ShapeSpec(channels=3, height=800, width=800), "resnet50")
        self.backbone2 = BandSpecificBackbone(cfg, ShapeSpec(channels=3, height=800, width=800), "resnet50")

        # Get the output shape of the intermediate feature maps from the first backbone, which will be used for fusion
        fusion_output_shape = self.backbone1.output_shape
        self.output_shape = {"0": fusion_output_shape[0]}

        # Define a convolutional layer for feature fusion
        # self.fusion = nn.Conv2d(fusion_output_shape[0].channels*2, fusion_output_shape[0].channels, kernel_size=1)
        self.fusion = nn.Conv2d(fusion_output_shape[0].channels * 2, fusion_output_shape[0].channels, kernel_size=1)

        # Define proposal generators and ROI heads for each band separately
        self.proposal_generator1 = build_proposal_generator(cfg, self.output_shape)
        self.proposal_generator2 = build_proposal_generator(cfg, self.output_shape)
        self.roi_heads1 = build_roi_heads(cfg, self.backbone1.output_shape)
        self.roi_heads2 = build_roi_heads(cfg, self.backbone2.output_shape)

        # Set the device for the model
        self.to(cfg.MODEL.DEVICE)

    def forward(self, batched_inputs):
        # Extract the two input images from the batch
        images1 = ImageList.from_tensors([x["input1"] for x in batched_inputs], self.backbone1.size_divisibility)
        images2 = ImageList.from_tensors([x["input2"] for x in batched_inputs], self.backbone2.size_divisibility)

        # Compute intermediate feature maps for each band separately
        features1 = self.backbone1(images1)
        features2 = self.backbone2(images2)

        # Fuse the intermediate feature maps from both backbones
        # fused_features = features1[0] + features2[0]  # Sum the first feature map from each backbone
        fused_features = torch.cat([features1[0], features2[0]], dim=1) #(N, 2*C, H, W)
        fused_features = self.fusion(fused_features)  # Apply the convolutional

        # Generate proposals for each band separately
        proposals1, proposal_losses1 = self.proposal_generator1(images1, [fused_features])
        proposals2, proposal_losses2 = self.proposal_generator2(images2, [fused_features])

        # Perform detection for each band separately
        # instances1, detector_losses1 = self.roi_heads1(features1, proposals1, batched_inputs[0]["instances1"])
        # instances2, detector_losses2 = self.roi_heads2(features2, proposals2, batched_inputs[0]["instances2"])
        instances1, detector_losses1 = self.roi_heads1(fused_features, proposals1, batched_inputs[0]["instances1"])
        instances2, detector_losses2 = self.roi_heads2(fused_features, proposals2, batched_inputs[0]["instances2"])

        # Calculate the losses
        losses = {}
        for name, loss in detector_losses1.items():
            losses["1_" + name] = loss
        for name, loss in detector_losses2.items():
            losses["2_" + name] = loss

        return instances1, instances2, losses





##################################################################################
############################## TESTING.PY ########################################
##################################################################################



import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from model import MultiBandRCNN

# create mock images for band_1 and band_2
band_1_image = np.random.rand(3, 800, 800) * 255
band_2_image = np.random.rand(3, 800, 800) * 255

# Create a mock dataset for the images
dataset_dicts = [
    {
        "file_name": "band_1.jpg",
        "image_id": 0,
        "height": 800,
        "width": 800,
        "image1": torch.as_tensor(band_1_image.transpose(2, 0, 1), dtype=torch.float32),
        "instances1": {
            "gt_boxes": torch.tensor([[100, 100, 400, 400]], dtype=torch.float32),
            "gt_classes": torch.tensor([0], dtype=torch.int64),
        },
        "image2": torch.as_tensor(band_2_image.transpose(2, 0, 1), dtype=torch.float32),
        "instances2": {
            "gt_boxes": torch.tensor([[200, 200, 500, 500]], dtype=torch.float32),
            "gt_classes": torch.tensor([1], dtype=torch.int64),
        },
    }
]

# Define a function to convert the mock dataset to the required format for Detectron2
def get_dataset_dicts():
    return dataset_dicts

# Register the mock dataset
register_coco_instances("my_dataset", {}, "my_dataset.json", get_dataset_dicts)

# Create the predictor
cfg = get_cfg()
cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = "model_final.pth"
predictor = DefaultPredictor(cfg)

# Create the MultiBandRCNN model
model = MultiBandRCNN(cfg)

# Perform detection on the mock images
outputs = model([
    {"input1": torch.as_tensor(band_1_image.transpose(2, 0, 1), dtype=torch.float32),
     "input2": torch.as_tensor(band_2_image.transpose(2, 0, 1), dtype=torch.float32)}
])

# Get the predicted instances for each band
instances1 = outputs[0].to("cpu")
instances2 = outputs[1].to("cpu")

# Visualize the predicted boxes for band_1
metadata = MetadataCatalog.get("my_dataset")
v = Visualizer(band_1_image[:, :, ::-1], metadata=metadata, scale=1.0)
v = v.draw_instance_predictions(instances1)
plt.imshow(v.get_image()[:, :, ::-1])
plt.show()

# Visualize the predicted boxes for band_2
v = Visualizer(band_2_image[:, :, ::-1], metadata=metadata, scale=1.0)
v = v.draw_instance_predictions(instances2)
plt.imshow(v.get_image()[:, :, ::-1])
plt.show()




##################################################################################
############################## TRAINING.PY #######################################
##################################################################################


import torch
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.data import DatasetMapper, build_detection_train_loader, build_detection_test_loader
from detectron2.structures import BoxMode
from detectron2.engine.defaults import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.checkpoint import PeriodicCheckpointer

from PIL import Image
import numpy as np

from models import MultiBandRCNN
from utils import build_optimizer, build_lr_scheduler





import torch
from detectron2.layers import ROIAlign

# class MultiBandedROIAlign(nn.Module):
#     '''this was not in original MLMT, but it would probably be good to try it. It is useful for pooling band specific
#        (e.g., pool from resnet_1 for band_1, from 2 for 2...etc., then concatenate the pooled features.) this could
#        (my instinct) preserve more band specific info that could be lost by reduction operations (e.g., Max/Min/Avg
#        pooling formulas)
#
#        One thing to work on (didn't have the time to do it): make sure you are performing ROIAlign for band_x using ground-truth of band_x.'''
#
#     def __init__(self, output_size, sampling_ratio):
#         super().__init__()
#         self.output_size = output_size
#         self.sampling_ratio = sampling_ratio
#
#     def forward(self, features, boxes_list):
#         # Separate the features by band
#         features_b1 = features[:, :, :features.shape[2] // 2, :]
#         features_b2 = features[:, :, features.shape[2] // 2:, :]
#
#         # Apply ROI align for each band
#         output_b1 = roi_align(features_b1, boxes_list[0], self.output_size, self.sampling_ratio)
#         output_b2 = roi_align(features_b2, boxes_list[1], self.output_size, self.sampling_ratio)
#
#         # Concatenate the outputs along the channel dimension
#         output = torch.cat([output_b1, output_b2], dim=1)
#
#         return output




class MultiBandedROIAlign(nn.Module):
    '''this was not in original MLMT, but it would probably be good to try it. It is useful for pooling band specific
       (e.g., pool from resnet_1 for band_1, from 2 for 2...etc., then concatenate the pooled features.) this could
       (my instinct) preserve more band specific info that could be lost by reduction operations (e.g., Max/Min/Avg
       pooling formulas)

       todo: chaeck symmantics: make sure you are performing ROIAlign for band_x using ground-truth of band_x, as
       opposing to band_x and band_y GT fused together'''

    def __init__(self, output_size, sampling_ratio):
        super().__init__()
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio

    def forward(self, features, boxes):
        # Separate the features by band
        features_b1 = features[:, :, :features.shape[2] // 2, :]
        features_b2 = features[:, :, features.shape[2] // 2:, :]

        # Apply ROI align for each band
        output_b1 = roi_align(features_b1, boxes, self.output_size, self.sampling_ratio)
        output_b2 = roi_align(features_b2, boxes, self.output_size, self.sampling_ratio)

        # Concatenate the outputs along the channel dimension
        output = torch.cat([output_b1, output_b2], dim=1)

        return output



# Define the custom dataset
def get_dataset():
    dataset_dicts = []

    for i in range(10):
        record = {}

        # Load the band-specific images
        band_1_image = np.random.rand(3, 800, 800) * 255
        band_2_image = np.random.rand(3, 800, 800) * 255

        # Add the images to the record
        record["file_name1"] = f"band1_{i}.jpg"
        record["image1"] = Image.fromarray(band_1_image.transpose(1, 2, 0).astype(np.uint8))
        record["file_name2"] = f"band2_{i}.jpg"
        record["image2"] = Image.fromarray(band_2_image.transpose(1, 2, 0).astype(np.uint8))

        # Add the annotations to the record
        annotations1 = []
        annotations2 = []

        num_objs = np.random.randint(1, 5)
        for j in range(num_objs):
            x1, y1, x2, y2 = np.random.randint(0, 800, size=4)
            annotations1.append({
                "bbox": [x1, y1, x2, y2],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": np.random.randint(1, 11)
            })
            annotations2.append({
                "bbox": [x1, y1, x2, y2],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": np.random.randint(1, 11)
            })

        record["annotations1"] = annotations1
        record["annotations2"] = annotations2

        dataset_dicts.append(record)

    return dataset_dicts

# Define the dataset name
dataset_name = "custom_dataset"

# Register the dataset with Detectron2
DatasetCatalog.register(dataset_name, get_dataset)
MetadataCatalog.get(dataset_name).set(thing_classes=[str(i) for i in range(1, 11)], num_classes=10)

# Set configuration parameters
cfg = get_cfg()
cfg.INPUT.MIN_SIZE_TRAIN = (800,)
cfg.INPUT.MAX_SIZE_TRAIN = 800
cfg.INPUT.MIN_SIZE_TEST = 800
cfg.INPUT.MAX_SIZE_TEST = 800
cfg.MODEL.MASK_ON = False
cfg.MODEL.RETINANET.NUM_CLASSES = 10
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
cfg.MODEL.ROI_HEADS.POOLER_RESOLUTION = 14
cfg.MODEL.ROI_HEADS.POOLER_SAMPLING_RATIO = 2
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
cfg.MODEL.ROI_HEADS.POOLER_NAME = "MultiBandedROIAlign"

# Create the MultiBandRCNN model
model = MultiBandRCNN(cfg)

# Build the optimizer
optimizer = build_optimizer(cfg, model)

# Build the lr_scheduler
lr_scheduler = build_lr_scheduler(cfg, optimizer)

# Define the trainer
trainer = DefaultTrainer(cfg)
trainer.build_train_loader(cfg)
trainer.build_test_loader(cfg)

# Train the model
trainer.resume_or_load(resume=False)
trainer.train()

# Evaluate the trained model on the test data
evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, dataset_name)
inference_on_dataset(model, val_loader, evaluator)
