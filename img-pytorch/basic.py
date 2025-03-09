# pytorch library
# torchvision : img transformations, pre-trained models, datasets
# bin classification: sigmoid
# multiclass: softmax

# CNN model: tensors => Conv2d => ReLU => MaxPool2d => Flatten => Linear=> Sigmoid/Softmax

# datasets foler structure (for bin class: cat vs dog)
# data/
#   train/
#       cat/
#          001.png
#       dog/
#          001.png
#   test/

from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import vgg16, VGG16_Weights
from torchvision.ops import nms
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
from torchvision.models import (resnet18, ResNet18_weights)
from PIL import Image
import torch.nn as nn
import torch
from torchvision.transforms import functional
from torchvision import datasets
import torchvision.transforms as transforms

train_dir = 'data/train'
train_dataset = ImageFolder(root=train_dir, transform=transforms.ToTensor())
classes = train_dataset.classes
print(classes)  # ['cat', 'dog']
print(train_dataset.class_to_idx)  # {'cat': 0, 'dog': 1}

# *********************** Binary image classification ***********************


class BinaryCNN (nn.Module):
    def __init__(self):
        super(BinaryCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16*112*112, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x

# ************ Multiclass img classification with CNN ***********


class MultiClassCNN(nn.Module):
    def __init__(self, num_classes):
        super(MultiClassCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x


# Conv2d: input channels
image = PIL.image.open('dog.png')
num_channels = functional.get_image_channels(image)
print(num_channels)

# Adding Conv layers


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)


conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                  kernel_size=3, stride=1, padding=1)
model = Net()
model.add_module('conv2', conv2)

print(model)  # print the cnn layers

print(model.conv2)  # print the conv2 layer

# Creating convolutional blocks : Stacking conv layers in a block with nn.Sequential()


class BinaryImageClassification(nn.Module):
    def __init__(self):
        super(BinaryImageClassification, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.conv_block(x)

# ************************** using a Pre-trained model **************************

# saving a complete pytorch model
# extensions include .pt or .pth


# save model weights with .state_dict()
torch.save(model.state_dict(), 'BinaryCNN.pth')

# instantiate a new model
new_model = BinaryCNN()

# load model weights with .load_state_dict()
new_model.load_state_dict(torch.load('BinaryCNN.pth'))

# downloading torchvision models

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
transforms = weights.transforms()

# prepare new input images
image = Image.open("cat013.jpg")
image_tensor = transform(image)
image_reshaped = image_tensor.unsqueeze(0)

# make predictions
model.eval()
with torch.no_grad():  # disable grad
    # pass img to model and remove batch dim
    pred = model(image_reshaped).squeeze(0)

pred_cls = pred.softmax(0)
# select highest prob class and extract its index
cls_id = pred_cls.argmax().item()
cls_name = weights.meta["categories"][cls_id]

print(f"Predicted class: {cls_name}")

# **************************** Bounding boxes **************************
# bbox coordinates: (xmin, ymin, xmax, ymax) -> top left corner, bottom right corner
# note image axis are: row moves from top to bottom, column moves from left to right on the image
# so , top left corner of image is at (row=0, col=0)

# converting pixels to tensors: ToTensor()
# tensor type: torch.float

# scaled tensor range: [0.0, 1.0]
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])
image_tensor = transform(image)

# converting pixels to tensors: PILToTensor()

# tensor type: torch.uint8 (unsigned 8-bit integer)
# unscaled tensor range: [0, 255]
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.PILToTensor()
])
image_tensor = transform(image)

# ********************** drawing bounding boxes **********************
bbox = torch.tensor([xmin, ymin, xmax, ymax])
bbox = bbox.unsqueeze(0)  # unsqueeze to two dimensions
bbox_image = draw_bounding_boxes(image_tensor, bbox, width=3, colors="red")

transform = transforms.Compose([
    transforms.ToPILImage()
])
pil_image = transform(bbox_image)
plt.imshow(pil_image)

# *********************** Evaluating object recognition models **********************
# classification and bbox localization
# Intersection over Union (IoU): ground truth, prediction, overlap area
# area of union = area of ground truth + area of prediction - area of overlap
# IOU = (area of overlap) / (area of union)

# IOU > 0.5 is good

# IOU in pytorch
bbox1 = [50, 50, 100, 100]
bbox2 = [75, 75, 125, 125]

bbox1 = torch.tensor(bbox1).unsqueeze(0)
bbox2 = torch.tensor(bbox2).unsqueeze(0)

iou = box_iou(bbox1, bbox2)
print(iou)

# predicting bounding boxes
model.eval()
with torch.no_grad():
    pred = model(input_image)
print(pred)

boxes = pred[0]['boxes']
labels = pred[0]['labels']
scores = pred[0]['scores']

# NMS (non-maximum suppression): a common technique to select the most confident bounding boxes
# non-max: discarding all boxes with low confidence score to contain an object
# suppression: discarding boxes with low IOU

box_indices = nms(
    boxes=boxes,        # boxes: tensor of shape (num_boxes, 4)
    scores=scores,
    iou_threshold=0.5
)
print(box_indices)
filtered_boxes = boxes[box_indices]

# **************** Object detection using R-CNN ****************
# Region based CNN family: RCNN (CNN -> selective search -> CNN)
# RCNN family: R-CNN, Fast-CNN, Faster-CNN
# RCNN (1. CNN -> 2. selective search -> 3. CNN)
# Module 1: CNN (generation of region proposals)
# Module 2: selective search ( feature extraction) (conv layers)
# Module 3: CNN (object detection, class and bbox pred)

vgg = vgg16(weights=VGG16_Weights.DEFAULT)
backbone = nn.Sequential(*list(vgg.features.children())
                         [:-1])  # * unwraps the list
input_dimension = nn.Sequential(
    *list(vgg.classifier.children()))[0].in_features
# create a new classifier
classifier = nn.Sequential(
    nn.Linear(input_dimension, 4096),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, 2)
)

box_regressor = nn.Sequential(
    nn.Linear(input_dimension, 4096),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, 4)
)

# Putting it all together : object detn model


class ObjectDetectorCNN(nn.Module):
    def __init__(self):
        super(ObjectDetectorCNN, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        self.backbone = nn.Sequential(
            *list(vgg.features.children())[:-1])  # * unwraps the list
        input_features = nn.Sequential(
            *list(vgg.classifier.children()))[0].in_features
        self.classifier = nn.Sequential(
            nn.Linear(input_features, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2)
        )
        self.box_regressor = nn.Sequential(
            nn.Linear(input_features, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4)
        )

    def forward(self, x):
        features = self.backbone(x)
        classes = self.classifier(features)
        bboxes = self.box_regressor(features)
        return bboxes, classes

# ************ Region network proposal with Faster R-CNN ************
# region:
# anchor box: predefined bounding box templates of different sizes and shapes
# backbone : CNN

# Region proposal networ (RPN)
# anchor generator: generates anchor boxes of different sizes and shapes
# classifier and regressor: predicts class and bbox for each anchor box
# RoI pooling: resize the RPN proposal to a fixed size for fully connected layers


# RPN in pytorch

anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128,),),
    aspect_ratios=((0.5, 1.0, 2.0),)
)
roi_pooler = MultiScaleRoIAlign(
    featmap_names=[0],
    output_size=7,
    sampling_ratio=2
)

# Fast R-CNN loss functions
# rpn_cls_criterion = nn.BCEWithLogitsLoss()
# rpn_reg_criterion = nn.MSELoss()
# rpn_cls_criterion = nn.CrossEntropyLoss() # multiple object classes
# rcnn_reg_criterion = nn.MSELoss()


backbone = torchvision.models.mobilenet_v2(weights='DEFAULT').features
backbone.out_channels = 1280

model = FasterRCNN(backbone=backbone, num_classes=2,
                   rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)

# load pre-trained faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')

num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# ********************** Image segmentation ************************
# each pixel in an image is assigned a class/label (all pixels belonging to the same class are treated equally)
# 3 types of segmentation:
# 1. semantic segmentation
# 2. instance segmentation
# 3. panoptic segmentation

# data annotations
image = Image.open('dog.png')
mask = Image.open('dog_mask.png')

transform = transforms.Compose([
    transforms.ToTensor()])

image_tensor = transform(image)
mask_tensor = transform(mask)

print(image_tensor.shape)
print(mask_tensor.shape)

mask_tensor.unique()

# creating a binary mask
binary_mask = torch.where(mask_tensor == 1/255,
                          torch.tensor(1.0), torch.tensor(0.0))

to_pil_image = transforms.ToPILImage()
mask = to_pil_image(binary_mask)
plt.imshow(mask)

# segmenting the object
object_tensor = image_tensor * binary_mask
to_pil_image = transforms.ToPILImage()
object_image = to_pil_image(object_tensor)

plt.imshow(object_image)

# Instance segmentation with Mask R-CNN

# pre-trained masked R-CNN in pytorch
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

image = Image.open('dog.png')
transform = transforms.Compose([

])
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    prediction = model(image_tensor)

prediction[0]["labels"]
prediction[0]["masks"]
prediction[0]["boxes"]
prediction[0]["scores"]

# displaying soft masks
masks = prediction[0]["masks"]
labels = prediction[0]["labels"]

for i in range(2):
    plt.imshow(image)
    plt.imshow(
        masks[i, 0],
        cmap="jet",
        alpha=0.5
    )
    plt.title(f"object: {class_names[labels[i]]}")
    plt.show()

# ********************** Semantic segmentation *******************
# no distinction between different instances of the same class
# useful for medical imaging or satellite image analysis
# popular architecture: U-Net

# Read U-Net architecture

# transposed convolution
upsample = nn.ConvTranspose2d(
    in_channels=64,
    out_channels=64,
    kernel_size=2,
    stride=2
)

# Encoder and decoder
# Encoder: convolutional blocks


def conv_block(self, in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


# U-Net layer definitions
class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()

        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(512, 256)
        self.dec2 = self.conv_block(256, 128)
        self.dec3 = self.conv_block(128, 64)
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))

        x = self.upconv3(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.dec1(x)

        x = self.upconv2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)

        x = self.upconv1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec3(x)

        x = self.out(x)
        return x


# running inference
model = Unet()
model.eval()

image = Image.open("car.png")
transform = transforms.Compose([
    transforms.ToTensor()
])
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    prediction = model(image_tensor).squeeze(0)

plt.imshow(prediction[1, :, :])
plt.show()

# Panoptic Segmentation (Read about it.)
