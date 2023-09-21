import numpy as np
import matplotlib.pyplot as plt
# import pdb

def calculate_endpoint(start_point, length, angle_rad):
    x_start, y_start = start_point
    delta_x = length * np.cos(angle_rad)
    delta_y = length * np.sin(angle_rad)
    x_end = x_start + delta_x
    y_end = y_start + delta_y
    return x_end, y_end

# select points whcih are atleast 5 pixels away from each other
def filter_points(points, x):
    filtered_points = []    
    for i, point1 in enumerate(points):
        is_valid = True
        
        for j, point2 in enumerate(points):
            if i != j:  # Avoid comparing the point to itself
                distance = np.linalg.norm(np.array(point1) - np.array(point2))
                if distance < x:
                    is_valid = False
                    break  # No need to check other points
                
        if is_valid:
            filtered_points.append(point1)
    # print(filtered_points)
    return np.array(filtered_points)
    
def visualize(image, obj_prob, star_distances, star_angles):
    # calculate 99.5 percentile object_probabilities
    image = np.array(image)
    obj_prob = np.array(obj_prob)
    star_distances = np.array(star_distances)
    star_angles = np.array(star_angles)
    percentile_99_5 = np.percentile(obj_prob, 99.5)
    # select points where object_probabilities > percentile_99.5
    mid_points = np.argwhere(obj_prob > percentile_99_5)
    # filter points
    filtered_points = filter_points(mid_points, 5)
    # pdb.set_trace()
    X = []
    Y = []
    for point in filtered_points:
        start_point = point
        length = star_distances[start_point[0], start_point[1]]
        angle_rad = star_angles[start_point[0], start_point[1]] * 2 * np.pi
        x , y = calculate_endpoint(start_point, length, angle_rad)
        X.append(x)
        Y.append(y)
    plt.imshow(image.transpose(1,2,0))
    plt.scatter(filtered_points[:,1], filtered_points[:,0], s=1, c='r')
    plt.scatter(Y,X, c='black', s = 1, marker='*')
    plt.title('Original image with star points')
    plt.show()
    plt.imshow(obj_prob, cmap='gray')
    plt.title('Object probabilities')
    plt.scatter(Y,X, c='y', s = 1, marker='*')
    plt.show()
   