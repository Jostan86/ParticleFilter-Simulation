import numpy as np
import math
from sim_functions import transform_map_to_rob
from sim_robot import Robot


class World:
    def __init__(self):
        # Create robot object
        self.robot = Robot()

        # Initialize the needed variables
        # Variables to track the robot post
        self.robot_x = 0
        self.robot_y = 0
        self.robot_angle = None
        self.robot_angle_prev = 0
        # Variables to track the tree parameters
        self.xs_trees = []
        self.ys_trees = []
        self.width_trees = []
        # Variables to track the path positions
        self.xs_path = []
        self.ys_path = []

        # Variable to track the current position of the robot along the path
        self.step_count = 0

        # Variables to record the sensor reading in reference to the map and the robot
        self.x_seen_tree_list_rob = []
        self.y_seen_tree_list_rob = []
        self.x_seen_tree_list_map = []
        self.y_seen_tree_list_map = []
        self.widths_of_sensed_trees = []

        # Standard deviation for tree width measurement
        self.tree_width_accuracy = 0.3

        # Set the default values for the simulation settings
        self.step_time_default = 50  # milli-sec
        self.speed_default = 3  # m/s
        self.turn_speed_default = np.pi/4  # rad/s - full turn in 8 seconds
        self.width_accuracy_default = 0.5  # idk
        self.field_of_view_default = 90  # rad
        self.sensor_range_default = 1.8 # meters
        self.location_sensor_accuracy_default = 0.1  # idk
        self.sample_rate_default = 10  # Hz
        self.movement_sensor_accuracy_default = 0.2  # idk
        self.num_rows_default = 5
        self.num_columns_default = 10
        self.tree_spacing_default = 1.5 # meters
        self.row_width_default = 2.58  # meters
        self.tree_spacing_noise_default = 0.2  # idk
        self.row_width_noise_default = 0.02

        # Set the settings to their default values
        self.step_time = self.step_time_default  # milli-sec
        self.speed = self.speed_default  # m/s
        self.turn_speed = self.turn_speed_default  # rad/s - full turn in 8 seconds
        self.width_accuracy = self.width_accuracy_default  # idk
        self.field_of_view = self.field_of_view_default  # rad
        self.sensor_range = self.sensor_range_default  # meters
        self.location_sensor_accuracy = self.location_sensor_accuracy_default  # idk
        self.sample_rate = self.sample_rate_default  # Hz
        self.movement_sensor_accuracy = self.movement_sensor_accuracy_default  # idk
        self.num_rows = self.num_rows_default
        self.num_columns = self.num_columns_default
        self.tree_spacing = self.tree_spacing_default # meters
        self.row_width = self.row_width_default  # meters
        self.tree_spacing_noise = self.tree_spacing_noise_default  # idk
        self.row_width_noise = self.row_width_noise_default

        # Set the range values for the settings
        self.step_time_range = (1, 1000)  # milli-sec
        self.speed_range = (1, 6)  # m/s
        self.turn_speed_range = (np.pi/10, np.pi)  # rad/s - full turn in 8 seconds
        self.width_accuracy_range = (0, 1) # idk
        self.field_of_view_range = (10, 170)  # rad
        self.sensor_range_range = (0.5, 10)  # meters
        self.location_sensor_accuracy_range = (0, 0.5)  # idk
        self.sample_rate_range = (1, 200)  # Hz
        self.movement_sensor_accuracy_range = (0, 1)  # idk
        self.num_rows_range = (2, 20)
        self.num_columns_range = (5, 100)
        self.tree_spacing_range = (0.25, 8) # meters
        self.row_width_range = (1, 5)
        self.tree_spacing_noise_range = (0, 0.5)  # idk
        self.row_width_noise_range = (0, 0.1)  # idk

        # Tell the robot the needed values
        self.robot.field_of_view = self.field_of_view
        self.robot.sensor_range = self.sensor_range
        self.robot.row_width = self.row_width
        self.robot.tree_spacing = self.tree_spacing
        self.robot.tree_width_accuracy = self.tree_width_accuracy

    def reset(self):
        # Remake the orchard and path
        self.make_orchard()
        self.make_path()

    def update_pose(self):
        # Update the robot position on the map based on the step count
        self.robot_x = self.xs_path[self.step_count]
        self.robot_y = self.ys_path[self.step_count]

        self.update_robot_angle()

    def make_orchard(self):
        # Set the x and y values for the trees in the orchard

        # Calculate the total number of trees
        num_trees = self.num_rows * self.num_columns

        # Reset the lists
        self.xs_trees = []
        self.ys_trees = []
        self.width_trees = []

        # For each tree set the x and y value and the width
        for i in range(num_trees):
            x_value = (i % self.num_columns) * self.tree_spacing + np.random.normal(0, self.tree_spacing_noise)
            self.xs_trees.append(x_value)
            y_value = self.row_width * math.floor(i / self.num_columns) + np.random.normal(0, self.row_width_noise)
            self.ys_trees.append(y_value)
            # Width is based on a normal distribution with a mean of 6.7 and SD of 1.1. Based on data for rows 97 and 98
            tree_width = np.random.normal(6.7, 1.1)
            self.width_trees.append(tree_width)

        # Give the robot the map
        self.robot.xs_trees = self.xs_trees
        self.robot.ys_trees = self.ys_trees
        self.robot.width_trees = self.width_trees

    def make_path(self):
        # Make the path. Basically the criteria was to weave back and forth through the rows and place steps depending
        # on the sample rate of the location sensor and the speed of the robot. These steps had to all be equal distant
        # apart, which is where things get a little tricky. The path is basically being hard coded.

        # Calculate the length of the row
        row_length = (self.num_columns-1) * self.tree_spacing

        # Set the initial robot position
        xs_path = [0]
        ys_path = [self.row_width/2]

        # Calculate the distance the robot will move per step
        distance_per_step = self.speed/self.sample_rate
        # Calculate how many radians the robot moves along the semi-circle when turning per meter of linear travel
        # angle_per_meter = np.pi/ (self.row_width/2 * np.pi)
        angle_per_meter = 2/self.row_width  # Same as above but above make more sense
        # Calculate how many radians it moves along the semi-circle per step
        angle_per_step = angle_per_meter * distance_per_step

        # Calculate the number of similar path chunks there are, where one is 2 straight sections and 2 curves
        num_repeats = math.ceil((self.num_rows-1)/2)

        # Make the path
        for k in range(num_repeats):

            j = k*2

            # Calculate the number of steps for the first straight section
            num_steps = math.floor((row_length-xs_path[-1]) / distance_per_step)
            # Calculate the x and y values for the robot positions along the first straight section
            for i in range(num_steps):
                # xs move forward each step
                xs_path.append(xs_path[-1] + distance_per_step)
                # ys are all the same for this section
                ys_path.append(self.row_width * j + self.row_width/2)

            # If there's an even number of rows and this is the final section, then the path is done
            if self.num_rows % 2 == 0 and k == num_repeats-1:
                break

            # Calculate how much is left in the straight section after the final position, and before the curve starts
            leftover = row_length - xs_path[-1]
            # Amount of the next step that will be on the curve
            curve_amount = distance_per_step - leftover
            # Calculate the angle along the curve the robot will be at after moving part of one step onto the curve
            angle_cur = (-np.pi / 2) + (curve_amount * angle_per_meter)
            # Calculate the number of steps that will be taken on the next curve section
            num_steps = math.floor((np.pi - curve_amount) / angle_per_step)
            # Set the next path x and y values depending on the row parameters and the current angle along the curve
            xs_path.append(row_length + (self.row_width / 2) * np.cos(angle_cur))
            ys_path.append(self.row_width * (j + 1) + (self.row_width / 2) * np.sin(angle_cur))
            # Move the robot along the curve
            for i in range(num_steps):
                angle_cur = angle_cur + angle_per_step
                xs_path.append(row_length + (self.row_width/2) * np.cos(angle_cur))
                ys_path.append((j+1) * self.row_width + (self.row_width/2) * np.sin(angle_cur))

            # Calculate the angle left on the curve
            angle_left = (np.pi / 2) - angle_cur
            # Calculate the distance left on the curve
            leftover = angle_left / angle_per_meter
            # Add the next position on the next straight section depending on the amount that was leftover on the curve
            xs_path.append(row_length - (distance_per_step - leftover))
            ys_path.append((j+1) * self.row_width + self.row_width / 2)

            # Calculate the number of steps on the next straight section
            num_steps = math.floor((row_length - (row_length-xs_path[-1])) / distance_per_step)

            # Add path positions along the straight section
            for i in range(num_steps):
                xs_path.append(xs_path[-1] - distance_per_step)
                ys_path.append((j+1) * self.row_width + self.row_width/2)

            # If there's an odd number of rows and this is the last section, then the path is done
            if self.num_rows % 2 == 1 and k == num_repeats-1:
                break

            # Calculate how much is left in the straight section after the final position, and before the curve starts
            leftover = xs_path[-1]
            # Amount of the next step that will be on the curve
            curve_amount = distance_per_step - leftover
            # Calculate the angle along the curve the robot will be at after moving part of one step onto the curve
            angle_cur = (np.pi / 2) - (curve_amount * angle_per_meter)
            # Calculate the number of steps that will be taken on the next curve section
            num_steps = math.floor((np.pi - curve_amount) / angle_per_step)
            # Set the next path x and y values depending on the row parameters and the current angle along the curve
            xs_path.append(0 - (self.row_width / 2) * np.cos(angle_cur))
            ys_path.append((j+2) * self.row_width + (self.row_width / 2) * -np.sin(angle_cur))
            # Move the robot along the curve
            for i in range(num_steps):
                angle_cur = angle_cur - angle_per_step
                xs_path.append(0 - (self.row_width / 2) * np.cos(angle_cur))
                ys_path.append((j+2) * self.row_width + (self.row_width/2) * -np.sin(angle_cur))

            # Calculate the angle left on the curve
            angle_left = (np.pi / 2) + angle_cur
            # Calculate the distance left on the curve
            leftover = angle_left / angle_per_meter
            # Add the next position on the next straight section depending on the amount that was leftover on the curve
            xs_path.append(0 + (distance_per_step - leftover))
            ys_path.append((j+2) * self.row_width + self.row_width / 2)

        # Save the path to the path varialbe
        self.xs_path = xs_path
        self.ys_path = ys_path

    def update_robot_angle(self):
        # Update the robot angle depending on where it is along the path. Really, just special cases at the beginning
        # and end of the path, otherwise it is the angle between the path positions before and after the robot's current
        # position
        if self.step_count == 0:
            dx = self.xs_path[1] - self.xs_path[0]
            dy = self.ys_path[1] - self.ys_path[0]
            self.robot_angle = math.atan2(dy, dx)
            self.robot_angle_prev = self.robot_angle
        elif self.step_count == len(self.xs_path)-1:
            self.robot_angle_prev = self.robot_angle
            dx = self.xs_path[self.step_count] - self.xs_path[self.step_count - 1]
            dy = self.ys_path[self.step_count] - self.ys_path[self.step_count - 1]
            self.robot_angle = math.atan2(dy, dx)
        else:
            self.robot_angle_prev = self.robot_angle
            dx = self.xs_path[self.step_count + 1] - self.xs_path[self.step_count - 1]
            dy = self.ys_path[self.step_count + 1] - self.ys_path[self.step_count - 1]
            self.robot_angle = math.atan2(dy, dx)

    def sensor_reading(self):
        # Do a sensor reading, finds trees in the robot sensor's field of view and returns them to the sim (relative to
        # the map) and the robot (relative to the robot)

        # Need to update robot sensor as if robot is at previous position, so that the particles are updated based on
        # sensor reading and then moved forward, then repeat
        self.robot.x_seen_tree_list_sensor = self.x_seen_tree_list_rob
        self.robot.y_seen_tree_list_sensor = self.y_seen_tree_list_rob
        self.robot.widths_of_sensed_trees = self.widths_of_sensed_trees

        # Array of tree locations and tree widths
        tree_locs = np.array([self.xs_trees, self.ys_trees, self.width_trees])

        # Narrow down the trees to check if the sensor can see based on their general proximity to the robot, takes all
        # trees in a square around the robot with side edges the length of the sensor range
        tree_locs_close = tree_locs[:, tree_locs[0, :] > self.robot_x - self.sensor_range]
        tree_locs_close = tree_locs_close[:, tree_locs_close[0, :] < self.robot_x + self.sensor_range]
        tree_locs_close = tree_locs_close[:, tree_locs_close[1, :] > self.robot_y - self.sensor_range]
        tree_locs_close = tree_locs_close[:, tree_locs_close[1, :] < self.robot_y + self.sensor_range]

        # Reset lists of seen trees
        self.x_seen_tree_list_rob = []
        self.y_seen_tree_list_rob = []
        self.x_seen_tree_list_map = []
        self.y_seen_tree_list_map = []
        self.widths_of_sensed_trees = []

        # If there are trees around the robot...
        if tree_locs_close.shape[1] > 0:

            # Loop through each tree and check if it's in the sensor's field of view
            for j in range(tree_locs_close.shape[1]):
                # Add noise to the tree position and width, this should maybe later, but hard to add later, and arguably
                # makes more sense here. In this case trees outside the FOV can be seen but will appear as on the FOV,
                # otherwise trees in the FOV could be seen as outside the FOV
                x_rel_map = tree_locs_close[0, j] + np.random.normal(0, self.location_sensor_accuracy)
                y_rel_map = tree_locs_close[1, j] + np.random.normal(0, self.location_sensor_accuracy)
                tree_width_sensor = tree_locs_close[2, j] + np.random.normal(0, self.tree_width_accuracy)

                # Find the tree position relative to the robot, if the forward direction is the +x axis and using a left
                # hand axis
                point_rel_rob = transform_map_to_rob(np.array([self.robot_x, self.robot_y]),
                                                     np.array([x_rel_map, y_rel_map]), self.robot_angle)
                # Record the x and y position relative to the robot
                x_rel_rob = point_rel_rob[0, 0]
                y_rel_rob = point_rel_rob[1, 0]

                # Calculate the angle of the tree relative to the robot
                tree_angle_rel_particle = math.atan2(y_rel_rob, x_rel_rob)
                tree_distance_from_particle = np.sqrt(y_rel_rob ** 2 + x_rel_rob ** 2)

                # If the tree is in the field of view of the robot, then record it as a seen tree
                if np.pi / 2 - np.radians(self.field_of_view) / 2 < tree_angle_rel_particle < np.pi / 2 + \
                        np.radians(self.field_of_view) / 2 and tree_distance_from_particle < self.sensor_range:

                    # Record the seen trees
                    self.x_seen_tree_list_rob.append(x_rel_rob)
                    self.y_seen_tree_list_rob.append(y_rel_rob)
                    self.widths_of_sensed_trees.append(tree_width_sensor)

                    self.x_seen_tree_list_map.append(x_rel_map)
                    self.y_seen_tree_list_map.append(y_rel_map)






