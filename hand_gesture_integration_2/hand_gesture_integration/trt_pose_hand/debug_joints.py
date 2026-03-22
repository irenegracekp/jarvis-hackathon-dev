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

PALM = 0
THUMB_1, THUMB_2, THUMB_3, THUMB_4 = 1, 2, 3, 4
INDEX_1, INDEX_2, INDEX_3, INDEX_4 = 5, 6, 7, 8
MIDDLE_1, MIDDLE_2, MIDDLE_3, MIDDLE_4 = 9, 10, 11, 12
RING_1, RING_2, RING_3, RING_4 = 13, 14, 15, 16
BABY_1, BABY_2, BABY_3, BABY_4 = 17, 18, 19, 20

NAMES = ["palm","thumb_1","thumb_2","thumb_3","thumb_4",
         "index_1","index_2","index_3","index_4",
         "middle_1","middle_2","middle_3","middle_4",
         "ring_1","ring_2","ring_3","ring_4",
         "baby_1","baby_2","baby_3","baby_4"]

with open('preprocess/hand_pose.json', 'r') as f:
    hand_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(hand_pose)
num_parts = len(hand_pose['keypoints'])
num_links = len(hand_pose['skeleton'])

WIDTH = 224
HEIGHT = 224

WEIGHTS = 'model/hand_pose_resnet18_baseline_att_224x224_A.pth'
model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
model.load_state_dict(torch.load(WEIGHTS, weights_only=False))

from trt_pose.parse_objects import ParseObjects
parse_objects = ParseObjects(topology, cmap_threshold=0.2, link_threshold=0.2)

from preprocessdata import preprocessdata
preprocess_data = preprocessdata(topology, num_parts)

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

def preprocess(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = PIL.Image.fromarray(image_rgb)
    image_tensor = transforms.functional.to_tensor(image_pil).cuda()
    image_tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image_tensor[None, ...]

def dist(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

cap = cv2.VideoCapture(10)

log = open('/home/orin/trt_pose_hand/joint_log.txt', 'w')
frame_num = 0

with torch.no_grad():
    while frame_num < 200:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (WIDTH, HEIGHT))
        data = preprocess(frame_resized)
        cmap, paf = model(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = parse_objects(cmap, paf)
        joints = preprocess_data.joints_inference(frame_resized, counts, objects, peaks)

        valid = sum(1 for j in joints if j != [0,0])
        if valid >= 10 and frame_num % 5 == 0:
            palm = joints[PALM]
            log.write(f"\n=== Frame {frame_num} valid={valid} ===\n")
            for i, (name, j) in enumerate(zip(NAMES, joints)):
                d = dist(j, palm) if j != [0,0] and palm != [0,0] else -1
                log.write(f"  {name:12s} = {j}  dist_from_palm={d:.1f}\n")

            # Per-finger analysis
            log.write("  FINGER ANALYSIS:\n")
            for fname, tip, mid, base in [
                ("thumb",  THUMB_4,  THUMB_2,  THUMB_1),
                ("index",  INDEX_4,  INDEX_2,  INDEX_1),
                ("middle", MIDDLE_4, MIDDLE_2, MIDDLE_1),
                ("ring",   RING_4,   RING_2,   RING_1),
                ("baby",   BABY_4,   BABY_2,   BABY_1),
            ]:
                t = joints[tip]
                m = joints[mid]
                b = joints[base]
                p = palm
                if t != [0,0] and m != [0,0] and p != [0,0]:
                    dt = dist(t, p)
                    dm = dist(m, p)
                    db = dist(b, p) if b != [0,0] else -1
                    log.write(f"    {fname:8s}: tip_d={dt:.1f} mid_d={dm:.1f} base_d={db:.1f}  tip>mid={dt>dm}  tip>base={dt>db if db>0 else '?'}\n")
                else:
                    log.write(f"    {fname:8s}: missing joints\n")
            log.flush()

        frame_num += 1

        # Show frame with joint count
        cv2.putText(frame_resized, f"frame:{frame_num} joints:{valid}", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        display = cv2.resize(frame_resized, (640, 640))
        cv2.imshow("Debug", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
log.close()
print("Done. Log saved to joint_log.txt")
