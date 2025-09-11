
import os
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2

TITLE = 'Palette SE + PE Test'
RESULT_PATH = 'experiments/test_Palette_scalar100_test_abl_traj_250910_035408/results/test/0' #test 


def predict_ssim(gt_img, pred_img):
    # gt_img = gt_img.point(lambda x: 0 if x < 128 else 255, '1')
    # pred_img = pred_img.point(lambda x: 0 if x < 128 else 255, '1')

    gt_np = np.array(gt_img)
    pred_np = np.array(pred_img)

    score = ssim(gt_np, pred_np, data_range=255)
    return score

def predict_iou(gt_img, pred_img, threshold=128):
    # gt_img = gt_img.point(lambda x: 0 if x < 128 else 255, '1')
    # pred_img = pred_img.point(lambda x: 0 if x < 128 else 255, '1')

    gt_np = np.array(gt_img)
    pred_np = np.array(pred_img)

    gt_bin = (gt_np >= threshold).astype(np.uint8)
    pred_bin = (pred_np >= threshold).astype(np.uint8)

    intersection = np.logical_and(gt_bin, pred_bin).sum()
    union = np.logical_or(gt_bin, pred_bin).sum()

    iou = intersection / union if union > 0 else 1.0
    return iou


def predict_accuracy(gt1, pred2):
    # threshold = 127
    gt1 = gt1.point(lambda x: 0 if x < 128 else 255, '1')
    pred2 = pred2.point(lambda x: 0 if x < 128 else 255, '1')
    # occupied_accuracy and free_accuracy
    occupied = 0
    free = 0
    gt_occupied = 0
    gt_free = 0
    for i in range(gt1.size[0]):
        for j in range(gt1.size[1]):
            if gt1.getpixel((i, j)) == 255:
                gt_occupied += 1
                if pred2.getpixel((i, j)) == 255:
                    occupied += 1
            else:
                gt_free += 1
                if pred2.getpixel((i, j)) == 0:
                    free += 1
    occupied_accuracy = occupied / gt_occupied if gt_occupied > 0 else 0
    free_accuracy = free / gt_free if gt_free > 0 else 0
    return occupied_accuracy, free_accuracy

def get_img(gt_path, pred_path):
    # check if path refers to same img
    gt_num = gt_path.split('_')[-1].split('.')[0]
    pred_num = pred_path.split('_')[-1].split('.')[0]
    if gt_num != pred_num:
        raise ValueError(f'Ground truth image {gt_num} does not match predicted image {pred_num}')
    # get images in the path of the directory
    gt_img_path = [i for i in os.listdir(gt_path) if i.endswith('.png')]
    pred_img_path = [i for i in os.listdir(pred_path) if i.endswith('.png')]
    # print('gt_img_path:', gt_img_path)
    # print('pred_img_path:', pred_img_path)
    if len(gt_img_path) != 1 or len(pred_img_path) != 1:
        raise ValueError(f'Expected one image in each directory, got {len(gt_img_path)} and {len(pred_img_path)}')
    gt_img = Image.open(os.path.join(gt_path, gt_img_path[0]))
    pred_img = Image.open(os.path.join(pred_path, pred_img_path[0]))
    gt_img = gt_img.convert('L')  # convert to grayscale
    pred_img = pred_img.convert('L')  # convert to grayscale

    return gt_img, pred_img



if __name__ == '__main__':
    dir_list = os.listdir(RESULT_PATH)
    # print('dir_list:', dir_list)

    GT_paths = []
    pred_paths = []

    for dir in dir_list:
        # split dir name with '_' and .png
        dir_type = dir.split('_')[0]
        if dir_type == 'GT':
            GT_paths.append(os.path.join(RESULT_PATH, dir))
        elif dir_type == 'Out':
            pred_paths.append(os.path.join(RESULT_PATH, dir))
    # print('GT_paths:', GT_paths)
    # print('pred_paths:', pred_paths)

    if len(GT_paths) != len(pred_paths):
        raise ValueError(f'Number of ground truth images {len(GT_paths)} does not match number of predicted images {len(pred_paths)}')
    else:
        print(f'Number of ground truth images: {len(GT_paths)}, Number of predicted images: {len(pred_paths)}')
    
    # average accuracy
    total_occupied_accuracy = 0
    total_free_accuracy = 0
    total_ssim = 0
    total_iou = 0
    for gt_path, pred_path in zip(GT_paths, pred_paths):
        gt_img, pred_img = get_img(gt_path, pred_path)
        # show images cv2
        # cv2.imshow('Ground Truth', np.array(gt_img))
        # cv2.imshow('Predicted', np.array(pred_img))
        # cv2.waitKey(0)
        
        occupied_accuracy, free_accuracy = predict_accuracy(gt_img, pred_img)
        ssim_score = predict_ssim(gt_img, pred_img)
        iou_score = predict_iou(gt_img, pred_img)
        # print(f'Ground truth: {gt_path}, Predicted: {pred_path}')
        # print(f'Occupied accuracy: {occupied_accuracy:.4f}, Free accuracy: {free_accuracy:.4f}, SSIM: {ssim_score:.4f}, IoU: {iou_score:.4f}')
        total_occupied_accuracy += occupied_accuracy
        total_free_accuracy += free_accuracy
        total_ssim += ssim_score
        total_iou += iou_score
        # break

    avg_occupied_accuracy = total_occupied_accuracy / len(GT_paths)
    avg_free_accuracy = total_free_accuracy / len(GT_paths)
    avg_ssim = total_ssim / len(GT_paths)
    avg_iou = total_iou / len(GT_paths)

    print(f'{TITLE}: \nAverage occupied accuracy: {avg_occupied_accuracy:.4f}, Average free accuracy: {avg_free_accuracy:.4f}, Average SSIM: {avg_ssim:.4f}, Average IoU: {avg_iou:.4f}')