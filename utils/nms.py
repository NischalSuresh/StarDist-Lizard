import numpy as np
import cv2 as cv


def calculate_endpoint(start_point, length, angle_rad):
    x_start, y_start = start_point[0], start_point[1]
    delta_x = length * np.cos(angle_rad)
    delta_y = length * np.sin(angle_rad)
    x_end = x_start + delta_x
    y_end = y_start + delta_y
    return x_end, y_end

def non_max_suppression(masks, iou_threshold):
    """
    Perform Non-Maximum Suppression (NMS) on a list of masks.
    
    Args:
        masks (numpy.ndarray): A numpy array of shape (n, 256, 256) containing binary masks.
        probabilities (list of floats): List of confidence probabilities for each mask (length n).
        iou_threshold (float): IoU (Intersection over Union) threshold for suppression.
        
    Returns:
        List of selected masks after NMS.
    """
    selected_masks = []  # List to store selected masks
    n = masks.shape[0]  # Number of masks

    # Iterate through masks in descending order of probability
    for i in range(n):
        mask = masks[i]
        keep_mask = True  # Flag to determine if the mask should be kept
        # Calculate IoU with previously selected masks
        size = 0
        for _ in selected_masks:
            size += 1
        for j in range(size):
            iou = np.sum(mask & selected_masks[j]) / np.sum(mask | selected_masks[j])
            
            # If IoU is above the threshold, suppress the current mask
            if iou > iou_threshold:
                keep_mask = False
                break
        
        # If the mask is not suppressed, add it to the selected masks list
        if keep_mask:
            selected_masks.append(mask)
    
    return selected_masks


def generate_masks(obj_prob, star_dist, angles, nms_threshold=0.2, min_prob=0.25):
    '''
    args:
    - obj_prob: object probability map, shape (1,1,H,W)
    - star_dist: star poly distance map, shape (1,n_rays,H,W)
    - angles: star poly angle map, shape (1,n_rays,H,W)
    - nms_threshold: nms threshold
    - min_prob: minimum probability threshold
    returns:
    - mask_nms: list of masks after nms
    '''
    obj_prob = obj_prob.squeeze().numpy()
    star_dist = star_dist.squeeze().permute(1,2,0).numpy()
    angles = angles.squeeze().permute(1,2,0).numpy()
    points = np.argwhere(obj_prob > min_prob)
    prob_points = obj_prob[points[:,0],points[:,1]]
    sorted_points = points[np.argsort(prob_points)[::-1]]
    num_points = sorted_points.shape[0]
    X = []
    Y = []
    for point in sorted_points:
        len = star_dist[point[0],point[1]]
        theta = angles[point[0],point[1]] * 2 * np.pi
        x , y = calculate_endpoint(point, len, theta)
        X.append(x)
        Y.append(y)
    X1 = np.array(X)
    Y1 = np.array(Y)
    end_points = np.stack((Y1,X1), axis=-1).astype(int)
    mask_init = np.zeros((num_points,256,256), dtype=np.uint8)
    for i in range(num_points):
        cv.fillPoly(mask_init[i], pts=[end_points[i]], color=1)
    mask_nms = non_max_suppression(mask_init, nms_threshold)
    return mask_nms


def visualize_masks(masks):
    """
    Visualize multiple masks with different colors superimposed on a blank image.

    Args:
        masks (list of numpy arrays): List of binary masks of shape (h, w).

    Returns:
        numpy.ndarray: The image with masks superimposed in different colors.
    """
    # Determine the image dimensions from the first mask
    h, w = masks[0].shape
    
    # Create an empty image with 3 channels (RGB)
    image = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Generate random colors for each instance
    colors = np.random.randint(0, 255, size=(len(masks), 3), dtype=np.uint8)
    
    # Superimpose each mask onto the image with a unique color
    for i, mask in enumerate(masks):
        color = colors[i]
        masked_image = cv.merge([mask * color[0], mask * color[1], mask * color[2]])
        image += masked_image
    
    return image
