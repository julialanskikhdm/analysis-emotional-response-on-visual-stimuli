import torch
import torchvision
import torchvision.models as models
from torchvision import transforms
import PIL
from PIL import Image
import facenet_pytorch
from facenet_pytorch import MTCNN
from hsemotion.facial_emotions import HSEmotionRecognizer
import cv2
from torchvision import transforms as trn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
import os

#@title Imports and function definitions

# For running inference on the TF-Hub module.
import tensorflow as tf

import tensorflow_hub as hub

# For downloading the image.
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

# For measuring the inference time.
import time

from ObjectClassification import *

main_model_path = "baseline_resnet18_classifier_for_app_v1.pth"

# будут другие пути на другие модели по предсказанию визуальных стимулов
main_model = torch.load(main_model_path, map_location=torch.device('cpu'))

emotions = {1: 'happy', 2: 'sad', 3: 'angry', 4: 'surprised'}
labels = {"emotion": {0: "Amusement", 1: "Awe", 2: "Contentment", 3: "Excitement", 4: "Anger", 5: "Disgust", 6: "Fear", 7: "Sadness"}}


def load_image_by_path(path):
  image = Image.open(path).convert('RGB')
  transform = transforms.Compose([transforms.Resize((224, 224)),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

  image = transform(image)
  return image

def predict_main(img_filepath):
  image_tensor = load_image_by_path(img_filepath)
  image_tensor = image_tensor.unsqueeze(0)
  main_model.eval()
  with torch.no_grad():
    answer = main_model(image_tensor)

    pred = F.softmax(answer.cpu())
    print(pred)
    emotion_distribution = {}

    for idx, emotion_name in labels["emotion"].items():
      probability = round(pred[0][idx].item(), 3)
      emotion_distribution[emotion_name] = probability
  return emotion_distribution



# другие функции на предсказание (predict) визуальных атрибутов
def detect_face(frame):
  mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device='cpu')
  bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)
  bounding_boxes = bounding_boxes[probs > 0.9]
  return bounding_boxes

def predict_face_expr(img_filepath):
  model_name = 'enet_b0_8_best_afew'
  fer = HSEmotionRecognizer(model_name=model_name, device='cpu')
  frame_bgr = cv2.imread(img_filepath)
  frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
  bounding_boxes = detect_face(frame)
  for bbox in bounding_boxes:
    box = bbox.astype(int)
    x1, y1, x2, y2 = box[0:4]
    face_img = frame[y1:y2, x1:x2, :]
    emotion, scores = fer.predict_emotions(face_img, logits=True)
    return emotion, scores

def predict_scene(img_filepath):
  # load the pre-trained weights
  model_file = 'resnet18_places365.pth.tar'
  arch = 'resnet18'
  model = models.__dict__[arch](num_classes=365)
  checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
  state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
  model.load_state_dict(state_dict)
  model.eval()

  # load the image transformer
  centre_crop = trn.Compose([
    trn.Resize((256, 256)),
    trn.CenterCrop(224),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])

  # load the class label
  file_name = 'categories_places365.txt'
  classes = list()
  with open(file_name) as class_file:
    for line in class_file:
      classes.append(line.strip().split(' ')[0][3:])
  classes = tuple(classes)

  # load the test image
  img = Image.open(img_filepath)
  input_img = V(centre_crop(img).unsqueeze(0))

  # forward pass
  logit = model.forward(input_img)
  print(logit)
  prediction = model(input_img)
  print(prediction)
  num = 0
  for i in prediction[0]:
    num += 1
  print(num)
  h_x = F.softmax(logit, 1).data.squeeze()
  probs, idx = h_x.sort(0, True)

  # print('{} prediction on {}'.format(arch, img_name))
  # output the prediction
  for i in range(0, 5):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
  return probs, idx, classes

def predict_objects(img_filepath):
  image = download_and_resize_image(img_filepath, 1280, 856)
  module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"  # @param ["https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1", "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"]
  detector = hub.load(module_handle).signatures['default']


  img = load_img(image)

  converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  start_time = time.time()
  result = detector(converted_img)
  end_time = time.time()

  result = {key: value.numpy() for key, value in result.items()}
  print(result)

  print("Found %d objects." % len(result["detection_scores"]))
  print("Inference time: ", end_time - start_time)

  image_with_boxes = draw_boxes(
    img.numpy(), result["detection_boxes"],
    result["detection_class_entities"], result["detection_scores"])

  display_image(image_with_boxes)

  detection_class_entities = result['detection_class_entities']
  unique_elements = list(set([elem.decode('utf-8') for elem in detection_class_entities]))
  return unique_elements


def predict_emotion_w_face_expr(img_filepath):
  model_name = 'enet_b0_8_best_afew'
  fer = HSEmotionRecognizer(model_name=model_name, device='cpu')
  features = []
  frame_bgr = cv2.imread(img_filepath)
  frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
  bounding_boxes = detect_face(frame)
  for bbox in bounding_boxes:
    if all(coord >= 0 for coord in bbox):
      print("я здесь")
      box = bbox.astype(int)
      x1, y1, x2, y2 = box[0:4]
      face_img = frame[y1:y2, x1:x2, :]
      features = fer.extract_features(face_img)
  x = torch.tensor(features)
  x = x.view(1, 1, 1280).type(torch.float32)
  x = x.repeat(1, 3, 1, 1)
  print(x.shape)
  model_for_analyze_face_expr_stimuli = torch.load("resnet18_fine-tune_hsemotionFE_bboxes.pth", map_location=torch.device('cpu'))
  model_for_analyze_face_expr_stimuli.eval()
  with torch.no_grad():

    answer = model_for_analyze_face_expr_stimuli(x)

    pred = F.softmax(answer.cpu())
    print(pred)
    emotion_distribution = {}

    for idx, emotion_name in labels["emotion"].items():
      probability = round(pred[0][idx].item(), 3)
      emotion_distribution[emotion_name] = probability
  return emotion_distribution

def predict_emotion_w_scene_type(img_filepath):
  new_model = torch.load("classification_withFE_Places365_FineTune_CNNMODEL.pth", map_location=torch.device('cpu'))
  arch = 'resnet18'
  model_file = '%s_places365.pth.tar' % arch
  if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)
  model = models.__dict__[arch](num_classes=365)
  checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
  state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
  model.load_state_dict(state_dict)
  model.fc = torch.nn.Identity()
  model.eval()

  # load the image transformer
  centre_crop = trn.Compose([
    trn.Resize((256, 256)),
    trn.CenterCrop(224),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])

  img = Image.open(img_filepath)
  input_img = V(centre_crop(img).unsqueeze(0))
  features = model(input_img)
  features = features.data.cpu().numpy()
  x = torch.tensor(features)
  x = x.view(1, 1, 512).type(torch.float32)
  x = x.repeat(1, 3, 1, 1)

  # new_model.to(device)
  with torch.no_grad():
    # answer = new_model(x.to(device))
    answer = new_model(x)
    pred = F.softmax(answer.cpu())
    print(pred)
    emotion_distribution = {}

    for idx, emotion_name in labels["emotion"].items():
      probability = round(pred[0][idx].item(), 3)
      emotion_distribution[emotion_name] = probability

  return emotion_distribution
