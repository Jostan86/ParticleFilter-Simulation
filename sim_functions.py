import numpy as np

def transform_map_to_rob(robot_on_map, point_on_map, robot_angle):
    """Transforms corordinatates from map (global) to the robot (local)
    robot_on_map: x,y position of robot on the map as numpy array
    robot_angle: angle of the robot on the map as numpy array
    point_on_map: x,y position of the point to be transformed on the map
    returns: x,y position of the point relative to the robot coordinate frame"""

    point_rel_map = np.array([[point_on_map[0]], [point_on_map[1]], [1]])

    robot_rel_map = np.array([[robot_on_map[0]], [robot_on_map[1]]])

    rotation_matrix = np.array(
        [[np.cos(robot_angle), -np.sin(robot_angle)],
         [np.sin(robot_angle), np.cos(robot_angle)]])

    rotation_matrix_inv = np.transpose(rotation_matrix)

    position_transformed = np.matmul(-1 * rotation_matrix_inv, robot_rel_map)

    transformation_matrix_inv = np.concatenate((rotation_matrix_inv, position_transformed), axis=1)
    transformation_matrix_inv = np.concatenate((transformation_matrix_inv, np.array([[0, 0, 1]])))

    return np.matmul(transformation_matrix_inv, point_rel_map)