import json
from pycocotools.coco import COCO
import os
import numpy as np
import cv2

gt_image = "/home/datasets/textGroup/ctw1500/instances_training.json"

def gen_boundary_info(gt_file):
    coco = COCO(gt_file)
    imgs_info = coco.imgs
    anns = coco.anns
    for idx, img_info in imgs_info.item():
        # img_id = img_info['id']
        file_name = img_info['file_name']
        ans_path_file_name = os.path.abspath(file_name)
        img = cv2.imread(ans_path_file_name)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        img_anns = coco.getAnnIds(img_id)
        for img_ann in img_anns:
            pass
    # for ann in anns:
    #     img_info = coco.loadImgs(ann['image_id'])image_id
    #     file_name = ing_info['file_name']
    #     abs_file_path = os.path.abspath(file_name)
    #     img = cv2.imread(abs_file_path)
    #     mask = np.zeros(img.shape[:2], dtype=np.uint8)

    pass

gen_boundary_info(gt_image)


