import numpy as np
import math
from sim_functions import transform_map_to_rob
import scipy.stats
# hi
class Robot:
    def __init__(self):

        # Initiate particle list
        self.particles = None

        # Set number of particles
        self.num_particle_default = 1000
        self.num_particle = self.num_particle_default
        self.num_particle_range = (100, 10000)  # particles

        # Set the SD for the noise added to the particle movement
        self.particle_movement_noise_default = 0.2
        self.particle_movement_noise = self.particle_movement_noise_default
        self.particle_movement_noise_range = (0, 1)

        # List to hold the tree positions on the map
        self.xs_trees = None
        self.ys_trees = None
        self.width_trees = None

        # List to store the positions of the seen trees relative to the map
        self.x_seen_tree_list_sensor = []
        self.y_seen_tree_list_sensor = []
        self.widths_of_sensed_trees = []

        # Create variable for robot's FOV and sensor range
        self.field_of_view = None  # rad
        self.sensor_range = None  # meters
        self.tree_spacing = None
        self.row_width = None
        self.tree_width_accuracy = None

        ## Some variables the robot could know but doesn't need to
        # self.location_sensor_accuracy = None  # idk
        # self.sample_rate = None  # Hz
        # self.movement_sensor_accuracy = None  # idk
        # self.num_rows = self.num_rows_default
        # self.num_columns = self.num_columns_default
        # self.tree_spacing = self.tree_spacing_default # meters
        # self.row_width = self.row_width_default  # meters
        # self.tree_spacing_noise = self.tree_spacing_noise_default  # idk
        # self.row_width_noise = self.row_width_noise_default
        # self.step_time = self.step_time_default  # milli-sec
        # self.speed = self.speed_default  # m/s
        # self.turn_speed = self.turn_speed_default  # rad/s - full turn in 8 seconds
    #edit
    def reset(self):
        # Reset the particles
        x_particles_center = self.tree_spacing * 1.5
        y_particles_center = self.row_width * 0.5
        x_particles_len = self.tree_spacing * 4
        y_particles_len = self.row_width * 1.5
        self.setup_particle_filter(x_particles_center, y_particles_center, x_particles_len, y_particles_len)

    def setup_particle_filter(self, x_center, y_center, x_len, y_len):
        # Sets up the particles
        self.particles = np.random.random((self.num_particle, 3))
        self.particles[:, 0] = self.particles[:, 0] * x_len + (x_center - x_len/2)
        self.particles[:, 1] = self.particles[:, 1] * y_len + (y_center - y_len/2)
        self.particles[:, 2] = self.particles[:, 2] * 2*np.pi

    def update_particle_filter(self, move_distance, angular_distance):
        """
        Update the particle filter based on a sensor reading, trees seen sensor updated in world sensor update function
        :param move_distance: Linear distance to move particles
        :param angular_distance: Angular distance to move particles
        :return:
        """
        # Setup list to store distance sums
        distance_sum_list = np.array([])
        # Record the tree locations and widths in an array
        tree_locs = np.array([self.xs_trees, self.ys_trees, self.width_trees])

        # Loop through each particle and find the trees it would see, then compare that to the trees the actual robot
        # sees and update the probability accordingly
        for i, (particle_x, particle_y, particle_angle) in enumerate(zip(self.particles[:, 0], self.particles[:, 1],
                                                                         self.particles[:, 2])):

            # Narrow down the trees to check if the particle can see based on their general proximity to the particle,
            # takes all trees in a square around the particle with side edges the length of the sensor range
            tree_locs_close = tree_locs[:, np.logical_and(tree_locs[0, :] > particle_x - self.sensor_range,
                                                          tree_locs[0, :] < particle_x + self.sensor_range)]
            tree_locs_close = tree_locs_close[:, np.logical_and(tree_locs_close[1, :] > particle_y - self.sensor_range,
                                                                tree_locs_close[1, :] < particle_y + self.sensor_range)]

            # Reset lists of seen trees
            x_rel_particle_list = []
            y_rel_particle_list = []
            widths_of_trees_particle_senses = []

            # Loop through each close tree and check if it's in the particle's sensor's field of view
            for x, y, width in zip(tree_locs_close[0, :], tree_locs_close[1, :], tree_locs_close[2, :]):

                # Find the tree position relative to the particle, if the forward direction is the +x axis and using
                # a left hand axis
                point_rel_particle = transform_map_to_rob(np.array([particle_x, particle_y]),
                                                               np.array([x, y]), particle_angle)

                # Record the x and y position relative to the particle
                x_rel_particle = point_rel_particle[0, 0]
                y_rel_particle = point_rel_particle[1, 0]

                # Calculate the angle of the tree relative to the particle
                tree_angle_rel_particle = math.atan2(y_rel_particle, x_rel_particle)
                tree_distance_from_particle = np.sqrt(y_rel_particle ** 2 + x_rel_particle ** 2)

                # If the tree is in the field of view of the particle's sensor, then record it as a seen tree
                if np.pi/2 - np.radians(self.field_of_view) / 2 < tree_angle_rel_particle < np.pi/2 + \
                        np.radians(self.field_of_view) / 2 and tree_distance_from_particle < self.sensor_range:
                    x_rel_particle_list.append(x_rel_particle)
                    y_rel_particle_list.append(y_rel_particle)
                    widths_of_trees_particle_senses.append(width)

            # Record the number of trees the particle sees and the number of trees the robot sees
            num_trees_particle_would_see = len(x_rel_particle_list)
            num_trees_sensed = len(self.x_seen_tree_list_sensor)

            # Reset the distance sum to 0, this is used as a proxy for the particle's probability, although a lower
            # value correlates to a higher probability
            distance_sum = 0

            # Factor to determine how much weight to give to unmatched trees and the min distance to consider a tree
            # seen by the particle and a tree seen by the robot to be a match
            weight_factor = 1

            # If neither the particle nor the robot saw a tree then the probability proxy is set to 2
            if num_trees_particle_would_see == 0 and num_trees_sensed == 0:
                distance_sum = 2
            # If only the robot or particle saw a tree, then the number of trees seen determines its probability
            elif num_trees_particle_would_see == 0:
                distance_sum = num_trees_sensed * weight_factor*2
            elif num_trees_sensed == 0:
                distance_sum = num_trees_particle_would_see * weight_factor * 2
            # If both the particle and the robot do both see a tree, then they have to be matched together and then
            # their probability depends on how far apart the matched trees are.
            # A weighting for the width should probably be added here
            else:
                xp_grid, xs_grid = np.meshgrid(x_rel_particle_list, self.x_seen_tree_list_sensor)
                yp_grid, ys_grid = np.meshgrid(y_rel_particle_list, self.y_seen_tree_list_sensor)
                wp_grid, ws_grid = np.meshgrid(widths_of_trees_particle_senses, self.widths_of_sensed_trees)
                distances = ((xp_grid - xs_grid) ** 2 + (yp_grid - ys_grid) ** 2) ** 0.5
                width_differences = wp_grid - ws_grid

                # for idw, width_difference in np.ndenumerate(width_differences):
                #     if abs(width_difference) > 2:
                #         distances[idw] = 4

                for p in range(min([num_trees_particle_would_see, num_trees_sensed])):
                    min_index = np.unravel_index(distances.argmin(), distances.shape)
                    distance_sum += distances[min_index]
                    distances[min_index[0], :] = 100
                    try:
                        distances[:, min_index[1]] = 100
                    except IndexError:
                        pass

                distance_sum += abs(num_trees_sensed - num_trees_particle_would_see)*weight_factor

            # Add this particle's distance to the list of all the distances
            distance_sum_list = np.append(distance_sum_list, distance_sum)

        # Invert the distance sum list so that higher values are bad, add a small number to give a baseline probability
        # and avoid dividing by zero
        # change jostan ~22
        # distance_sum_list2 = 1 / (distance_sum_list + 0.00001)
        # This inversion formula seems to make convergence more likely, but a bit slower, and the spread is larger
        distance_sum_list2 = 4 / (distance_sum_list + 0.1)
        # distance_sum_list2 = 1 / distance_sum_list

        # Normalize the probabilities
        probabilites = distance_sum_list2 * (1 / distance_sum_list2.sum())

        # Make the list
        probabilites_sum = 0

        rand_values = np.sort(np.random.rand(self.num_particle))

        # Save the old particles
        old_particles = np.copy(self.particles)
        p_count = 0

        # Resample and move the particles
        for i, (probability, old_particle) in enumerate(zip(probabilites, old_particles)):
            probabilites_sum += probability

            while rand_values[p_count] < probabilites_sum and p_count < self.num_particle-1:

                self.particles[p_count, :] = old_particle

                self.particles[p_count, 2] = self.particles[p_count, 2] + np.random.normal(angular_distance, np.radians(5))
                self.particles[p_count, 0] = self.particles[p_count, 0] + np.cos(self.particles[p_count, 2]) * \
                                            np.random.normal(move_distance, move_distance * self.particle_movement_noise)
                self.particles[p_count, 1] = self.particles[p_count, 1] + np.sin(self.particles[p_count, 2]) * \
                                            np.random.normal(move_distance, move_distance * self.particle_movement_noise)

                p_count += 1







