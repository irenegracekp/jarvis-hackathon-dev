import json
import cv2
import trt_pose.coco
import trt_pose.models
import torch
import torchvision.transforms as transforms
import PIL.Image
import numpy as np
import os
import sys
import time
import pickle

with open('preprocess/hand_pose.json', 'r') as f:
    hand_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(hand_pose)
num_parts = len(hand_pose['keypoints'])
num_links = len(hand_pose['skeleton'])

WIDTH = 224
HEIGHT = 224

# Load pose model
WEIGHTS = 'model/hand_pose_resnet18_baseline_att_224x224_A.pth'
print("Loading hand pose model...", flush=True)
model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
model.load_state_dict(torch.load(WEIGHTS, weights_only=False))
print("Model loaded.", flush=True)

from trt_pose.parse_objects import ParseObjects
parse_objects = ParseObjects(topology, cmap_threshold=0.12, link_threshold=0.15)

from preprocessdata import preprocessdata
preprocessdata = preprocessdata(topology, num_parts)

# Load SVM gesture classifier
print("Loading SVM gesture classifier...", flush=True)
clf = pickle.load(open('svmmodel.sav', 'rb'))

with open('preprocess/gesture.json', 'r') as f:
    gesture = json.load(f)
gesture_type = gesture["classes"]
print(f"Gesture classes: {gesture_type}", flush=True)

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()


def preprocess(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = PIL.Image.fromarray(image_rgb)
    image_tensor = transforms.functional.to_tensor(image_pil).cuda()
    image_tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image_tensor[None, ...]


def draw_joints(image, joints):
    count = 0
    for i in joints:
        if i == [0, 0]:
            count += 1
    if count >= 3:
        return
    for i in joints:
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 1)
    cv2.circle(image, (joints[0][0], joints[0][1]), 2, (255, 0, 255), 1)
    for i in hand_pose['skeleton']:
        if joints[i[0]-1][0] == 0 or joints[i[1]-1][0] == 0:
            break
        cv2.line(image, (joints[i[0]-1][0], joints[i[0]-1][1]),
                 (joints[i[1]-1][0], joints[i[1]-1][1]), (0, 255, 0), 1)


GESTURE_COLORS = {
    "fist": (128, 128, 255),
    "pan": (0, 255, 255),
    "stop": (0, 0, 255),
    "peace": (255, 255, 0),
    "ok": (0, 255, 0),
    "no hand": (100, 100, 100),
}

CAMERA_DEVICE = 10
cap = cv2.VideoCapture(CAMERA_DEVICE)
if not cap.isOpened():
    print(f"Error: Cannot open camera /dev/video{CAMERA_DEVICE}", flush=True)
    sys.exit(1)

print(f"Camera opened. Gestures: {', '.join(gesture_type)}", flush=True)
print("Press 'q' to quit", flush=True)

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (WIDTH, HEIGHT))

        t0 = time.time()
        data = preprocess(frame_resized)
        cmap, paf = model(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = parse_objects(cmap, paf)
        joints = preprocessdata.joints_inference(frame_resized, counts, objects, peaks)
        dt = time.time() - t0

        # Draw skeleton
        draw_joints(frame_resized, joints)

        # SVM gesture classification
        dist_bn_joints = preprocessdata.find_distance(joints)
        gesture_pred = clf.predict([dist_bn_joints, [0] * num_parts * num_parts])
        gesture_joints = gesture_pred[0]
        preprocessdata.prev_queue.append(gesture_joints)
        preprocessdata.prev_queue.pop(0)

        # Get stabilized label
        preprocessdata.print_label(frame_resized, preprocessdata.prev_queue, gesture_type)
        current_gesture = preprocessdata.text

        # HUD
        fps = 1.0 / dt if dt > 0 else 0
        cv2.putText(frame_resized, f"FPS: {fps:.1f}", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Show gesture at bottom
        color = GESTURE_COLORS.get(current_gesture, (255, 255, 255))
        cv2.putText(frame_resized, current_gesture, (5, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        display = cv2.resize(frame_resized, (640, 640))
        cv2.imshow("Hand Pose + Gesture", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
