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
        self.width_seen_trees = []

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
            # Record particle pose
            # particle_x = self.particles[i, 0]
            # particle_y = self.particles[i, 1]
            # particle_angle = self.particles[i, 2]

            # Narrow down the trees to check if the particle can see based on their general proximity to the particle,
            # takes all trees in a square around the particle with side edges the length of the sensor range
            tree_locs_close = tree_locs[:, np.logical_and(tree_locs[0, :] > particle_x - self.sensor_range,
                                                          tree_locs[0, :] < particle_x + self.sensor_range)]
            tree_locs_close = tree_locs_close[:, np.logical_and(tree_locs_close[1, :] > particle_y - self.sensor_range,
                                                                tree_locs_close[1, :] < particle_y + self.sensor_range)]

            # Reset lists of seen trees
            x_rel_particle_list = []
            y_rel_particle_list = []
            width_trees_particle_sees = []

            # Loop through each tree and check if it's in the particle's sensor's field of view
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
                    width_trees_particle_sees.append(width)

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

                j_full, k_full = np.meshgrid(np.arange(len(x_rel_particle_list)), np.arange(len(self.x_seen_tree_list_sensor)))
                distance = np.zeros(j_full.shape)
                for j, k in zip(np.nditer(j_full), np.nditer(k_full)):
                    distance[k, j] = math.sqrt((x_rel_particle_list[j] - self.x_seen_tree_list_sensor[k]) ** 2 +
                                               (y_rel_particle_list[j] - self.y_seen_tree_list_sensor[k]) ** 2)

                for p in range(min([num_trees_particle_would_see, num_trees_sensed])):
                    min_index = np.unravel_index(distance.argmin(), distance.shape)
                    try:
                        distance_sum += distance[min_index[0], min_index[1]]
                    except IndexError:
                        distance_sum += distance[min_index]
                    distance[min_index[0], :] = 100
                    try:
                        distance[:, min_index[1]] = 100
                    except IndexError:
                        pass

                #         # z_score = abs((width_trees_particle_sees[j] - self.width_seen_trees[k]) / self.tree_width_accuracy)
                #         # width_weight = scipy.stats.norm.sf(z_score)
                #         # distance = distance * (1-width_weight)
                #         # distance = distance * z_score/5

                distance_sum += abs(num_trees_sensed - num_trees_particle_would_see)*weight_factor

            # Add this particle's distance to the list of all the distances
            distance_sum_list = np.append(distance_sum_list, distance_sum)

        # Invert the distance sum list so that higher values are bad, add a small number to give a baseline probability
        # and avoid dividing by zero
        distance_sum_list2 = 1 / (distance_sum_list + 0.00001)
        # distance_sum_list2 = 1 / distance_sum_list

        # Normalize the probabilities
        probabilites = distance_sum_list2 * (1 / distance_sum_list2.sum())

        # Make the list
        probabilites_sum = 0

        rand_values = np.sort(np.random.rand(self.num_particle))

        # Make an array of the new particles
        new_particles = np.zeros(self.particles.shape)

        p_cnt = 0
        for i, (probability, old_particle) in enumerate(zip(probabilites, self.particles)):
            probabilites_sum += probability

            # probabilites_histo[i] = probabilites_sum
            while rand_values[p_cnt] < probabilites_sum and p_cnt < self.num_particle-1:

                new_particles[p_cnt, :] = old_particle

                new_particles[p_cnt, 2] = new_particles[p_cnt, 2] + np.random.normal(angular_distance, np.radians(5))
                new_particles[p_cnt, 0] = new_particles[p_cnt, 0] + np.cos(new_particles[p_cnt, 2]) * \
                                          np.random.normal(move_distance, move_distance * self.particle_movement_noise)
                new_particles[p_cnt, 1] = new_particles[p_cnt, 1] + np.sin(new_particles[p_cnt, 2]) * \
                                          np.random.normal(move_distance, move_distance * self.particle_movement_noise)

                p_cnt += 1


        self.particles = new_particles[:]




