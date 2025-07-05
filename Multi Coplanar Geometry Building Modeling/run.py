import argparse
import os
from os.path import join

import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt

from gluestick import batch_to_np, numpy_image_to_torch, GLUESTICK_ROOT
from .models.two_view_pipeline import TwoViewPipeline
from .drawing import plot_images, plot_lines, plot_color_line_matches, plot_keypoints, plot_matches


def enhance_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_image


def compute_ncm_accuracy(matched_kps0, matched_kps1, H):
    transformed_kps1 = cv2.perspectiveTransform(matched_kps1[None, :, :], H)[0]
    distances = np.linalg.norm(matched_kps0 - transformed_kps1, axis=1)
    mean_distance = np.mean(distances)
    return mean_distance


def create_image_pyramid(image, num_levels):
    pyramid = [image]
    for i in range(1, num_levels):
        scaled_image = cv2.pyrDown(pyramid[-1])
        pyramid.append(scaled_image)
    return pyramid


def extract_edges(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    return edges


def non_maximum_suppression(lines, threshold=30):
    if len(lines) == 0:
        return []

    def line_length(line):
        x1, y1, x2, y2 = line
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    lines = sorted(lines, key=line_length, reverse=True)
    suppressed = [lines[0]]
    for line in lines[1:]:
        add = True
        for s_line in suppressed:
            if abs(line_length(line) - line_length(s_line)) < threshold:
                add = False
                break
        if add:
            suppressed.append(line)
    return suppressed


def probabilistic_hough_transform(edges, min_line_length=50, max_line_gap=10, num_votes=5):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=num_votes,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    if lines is not None:
        lines = lines.reshape(-1, 4)
        lines = non_maximum_suppression(lines)
    else:
        lines = np.empty((0, 4), dtype=int)
    return lines


def weighted_hough_transform(edges, edge_confidences, min_line_length=50, max_line_gap=10, num_votes=5):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=num_votes,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    if lines is not None:
        lines = lines.reshape(-1, 4)
        lines = non_maximum_suppression(lines)
    else:
        lines = np.empty((0, 4), dtype=int)
    return lines


def main():
    parser = argparse.ArgumentParser(
        prog='GlueStick Demo',
        description='Demo app to show the point and line matches obtained by GlueStick')
    parser.add_argument('-img1', default=join('resources', 'img1.jpg'))
    parser.add_argument('-img2', default=join('resources', 'img2.jpg'))
    parser.add_argument('--max_pts', type=int, default=1000)
    parser.add_argument('--max_lines', type=int, default=300)
    parser.add_argument('--skip-imshow', default=False, action='store_true')
    args = parser.parse_args()

    conf = {
        'name': 'two_view_pipeline',
        'use_lines': True,
        'extractor': {
            'name': 'wireframe',
            'sp_params': {
                'force_num_keypoints': False,
                'max_num_keypoints': args.max_pts,
            },
            'wireframe_params': {
                'merge_points': True,
                'merge_line_endpoints': True,
            },
            'max_n_lines': args.max_lines,
        },
        'matcher': {
            'name': 'gluestick',
            'weights': str(GLUESTICK_ROOT / 'resources' / 'weights' / 'checkpoint_GlueStick_MD.tar'),
            'trainable': False,
        },
        'ground_truth': {
            'from_pose_depth': False,
        }
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    base_extractor = TwoViewPipeline(conf).extractor
    pipeline_model = TwoViewPipeline(conf).to(device).eval()

    img0 = cv2.imread(args.img1)
    img1 = cv2.imread(args.img2)

    enhanced_img0 = enhance_image(img0)
    enhanced_img1 = enhance_image(img1)

    gray0 = cv2.cvtColor(enhanced_img0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(enhanced_img1, cv2.COLOR_BGR2GRAY)

    edges0 = extract_edges(enhanced_img0)
    edges1 = extract_edges(enhanced_img1)

    edge_confidences0 = cv2.normalize(edges0, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    edge_confidences1 = cv2.normalize(edges1, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    lines0 = weighted_hough_transform(edges0, edge_confidences0)
    lines1 = weighted_hough_transform(edges1, edge_confidences1)

    num_pyramid_levels = 3
    pyramid0 = create_image_pyramid(gray0, num_pyramid_levels)
    pyramid1 = create_image_pyramid(gray1, num_pyramid_levels)

    all_matched_kps0 = []
    all_matched_kps1 = []

    for level in range(num_pyramid_levels):
        torch_gray0, torch_gray1 = numpy_image_to_torch(pyramid0[level]), numpy_image_to_torch(pyramid1[level])
        torch_gray0, torch_gray1 = torch_gray0.to(device)[None], torch_gray1.to(device)[None]
        x = {'image0': torch_gray0, 'image1': torch_gray1}
        pred = pipeline_model(x)

        pred = batch_to_np(pred)
        kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
        m0 = pred["matches0"]

        valid_matches = m0 != -1
        match_indices = m0[valid_matches]
        matched_kps0 = kp0[valid_matches] * (2 ** level)
        matched_kps1 = kp1[match_indices] * (2 ** level)

        all_matched_kps0.append(matched_kps0)
        all_matched_kps1.append(matched_kps1)

    all_matched_kps0 = np.concatenate(all_matched_kps0, axis=0)
    all_matched_kps1 = np.concatenate(all_matched_kps1, axis=0)

    H, mask = cv2.findHomography(all_matched_kps1, all_matched_kps0, cv2.RANSAC)
    h, w = img0.shape[:2]

    # 透视变换后的图像
    img1_transformed = cv2.warpPerspective(enhanced_img1, H, (w, h))

    # 创建白色背景
    white_background = np.ones_like(img1_transformed) * 255

    # 找到非黑色的像素位置
    non_black_pixels_mask = np.all(img1_transformed != [0, 0, 0], axis=-1)

    # 将非黑色像素复制到白色背景上
    white_background[non_black_pixels_mask] = img1_transformed[non_black_pixels_mask]

    # 将透视变换后的图像保存
    cv2.imwrite('img1_transformed_with_white_bg.jpg', white_background)

    img1_restored = cv2.resize(white_background, (w, h))

    gray0_restored = cv2.cvtColor(enhanced_img0, cv2.COLOR_BGR2GRAY)
    gray1_restored = cv2.cvtColor(img1_restored, cv2.COLOR_BGR2GRAY)

    torch_gray0_restored = numpy_image_to_torch(gray0_restored).to(device)[None]
    torch_gray1_restored = numpy_image_to_torch(gray1_restored).to(device)[None]

    x_restored = {'image0': torch_gray0_restored, 'image1': torch_gray1_restored}
    pred_restored = pipeline_model(x_restored)

    pred_restored = batch_to_np(pred_restored)
    kp0_restored, kp1_restored = pred_restored["keypoints0"], pred_restored["keypoints1"]
    m0_restored = pred_restored["matches0"]

    line_seg0_restored, line_seg1_restored = pred_restored["lines0"], pred_restored["lines1"]
    line_matches_restored = pred_restored["line_matches0"]

    valid_matches_restored = m0_restored != -1
    match_indices_restored = m0_restored[valid_matches_restored]
    matched_kps0_restored = kp0_restored[valid_matches_restored]
    matched_kps1_restored = kp1_restored[match_indices_restored]

    valid_matches_restored = line_matches_restored != -1
    match_indices_restored = line_matches_restored[valid_matches_restored]
    matched_lines0_restored = line_seg0_restored[valid_matches_restored]
    matched_lines1_restored = line_seg1_restored[match_indices_restored]

    ncm_accuracy_restored = compute_ncm_accuracy(matched_kps0_restored, matched_kps1_restored, H)
    print(f'NCM Accuracy for restored image: {ncm_accuracy_restored}')
    print(f'Number of matched keypoints: {len(matched_kps0_restored)}')
    print(f'Number of matched lines: {len(matched_lines0_restored)}')

    enhanced_img0, img1_restored = cv2.cvtColor(gray0, cv2.COLOR_GRAY2BGR), cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)
    img0, img1 = cv2.cvtColor(gray0, cv2.COLOR_GRAY2BGR), cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)
    plot_images([enhanced_img0, img1_restored], ['Image 1 - detected lines', 'Restored Image - detected lines'],
                dpi=200,
                pad=2.0)
    plot_lines([line_seg0_restored, line_seg1_restored], ps=4, lw=2)
    plt.gcf().canvas.manager.set_window_title('Detected Lines')
    plt.savefig('detected_lines_restored.png')

    plot_images([enhanced_img0, img1_restored], ['Image 1 - detected points', 'Restored Image - detected points'],
                dpi=200,
                pad=2.0)
    plot_keypoints([kp0_restored, kp1_restored], colors='c')
    plt.gcf().canvas.manager.set_window_title('Detected Points')
    plt.savefig('detected_points_restored.png')

    plot_images([enhanced_img0, img1_restored], ['Image 1 - line matches', 'Restored Image - line matches'], dpi=200,
                pad=2.0)
    plot_color_line_matches([matched_lines0_restored, matched_lines1_restored], lw=2)
    plt.gcf().canvas.manager.set_window_title('Line Matches')
    plt.savefig('line_matches_restored.png')

    plot_images([enhanced_img0, img1_restored], ['Image 1 - point matches', 'Restored Image - point matches'], dpi=200,
                pad=2.0)
    plot_matches(matched_kps0_restored, matched_kps1_restored, 'green', lw=1, ps=0)
    plt.gcf().canvas.manager.set_window_title('Point Matches')
    plt.savefig('point_matches_restored.png')

    matched_kps1_restored_back = cv2.perspectiveTransform(matched_kps1_restored[None, :, :], np.linalg.inv(H))[0]
    matched_lines1_restored_back = cv2.perspectiveTransform(matched_lines1_restored.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)

    plot_images([img0, img1_restored], ['Image 1 - detected points', 'Restored Image - detected points'], dpi=200,
                pad=2.0)
    plot_keypoints([matched_kps0_restored, matched_kps1_restored], colors='c')
    plt.gcf().canvas.manager.set_window_title('Detected Points on Restored Image')
    plt.savefig('detected_points_on_restored.png')

    plot_images([img0, img1], ['Image 1 - point matches', 'Original Image - point matches'], dpi=200, pad=2.0)
    plot_matches(matched_kps0_restored, matched_kps1_restored_back, 'green', lw=1, ps=0)
    plt.gcf().canvas.manager.set_window_title('Point Matches on Original Image')
    plt.savefig('point_matches_on_original.png')

    plot_images([img0, img1], ['Image 1 - line matches', 'Original Image - line matches'], dpi=200, pad=2.0)
    plot_color_line_matches([matched_lines0_restored, matched_lines1_restored_back.reshape(-1, 2, 2)], lw=2)
    plt.gcf().canvas.manager.set_window_title('Line Matches on Original Image')
    plt.savefig('line_matches_on_original.png')

    grid_height = h // 3
    grid_width = w // 4

    for i in range(3):
        for j in range(4):
            y_start = i * grid_height
            y_end = (i + 1) * grid_height
            x_start = j * grid_width
            x_end = (j + 1) * grid_width
            grid_cell = img1_restored[y_start:y_end, x_start:x_end]
            cv2.imwrite(f'grid_cell_{i}_{j}_restored.png', grid_cell)

    if not args.skip_imshow:
        plt.show()


if __name__ == '__main__':
    main()
