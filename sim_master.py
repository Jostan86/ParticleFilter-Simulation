import sys
import matplotlib
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QSpinBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.figure import Figure
from sim_world import World
matplotlib.use('Qt5Agg')


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Initiate world object
        self.world = World()

        # Initiate references and variable for plotting
        self.plot2_ref_seen_point = None
        self.rect = None
        self.triangle = None
        self.arrow = None
        self.sensor_FOV_plot = None

        self.rect_x_size = 8
        self.rect_y_size = 5
        self.ratio_value = None
        self.rect_x_loc = None
        self.rect_y_loc = None

        self.plot1_ref_PF = None
        self.plot2_ref_PF = None

        # Boolean for entering a paused state
        self.pause = False

        # Create QT window
        self.window = QWidget()

        # Create overall layout and layout for the controls
        self.layout_overall = QHBoxLayout()
        self.layout_controls = QVBoxLayout()

        # Create controls
        self.label_heading = QLabel('Simulation Playback Settings')
        self.label_heading.setFont(QFont('Arial', 16))
        self.layout_controls.addWidget(self.label_heading)

        self.label_mode_choice = QLabel('Choose a Mode:')
        self.label_mode_choice.setMaximumWidth(130)
        mode_choice_toolTip = ("Choose a playback mode for the robot. During continuous play mode the robot will move "
                               "forward along the path automatically. It will move with the speed set below, the "
                               "particle filter will update based on the refresh rate of the tree location sensor. "
                               "During step mode the user can control when each step is take. During control mode the "
                               "user control where the robot moves and when the sensor is read. The particle filter "
                               "updates for each sensor read and each movement.")
        self.label_mode_choice.setToolTip(mode_choice_toolTip)
        spin_box_mode_choice_string = ["Continuous Play", "Step Mode", "Control Mode"]
        self.spinBox_mode_choice = StringBox(spin_box_mode_choice_string)
        self.spinBox_mode_choice.setToolTip(mode_choice_toolTip)
        self.spinBox_mode_choice.setWrapping(True)
        self.spinBox_mode_choice.valueChanged.connect(self.upon_spinbox_mode_choice_value_change)
        self.layout_spinBox = QHBoxLayout()
        self.layout_spinBox.addWidget(self.label_mode_choice)
        self.layout_spinBox.addWidget(self.spinBox_mode_choice)
        self.layout_controls.addLayout(self.layout_spinBox)

        self.button_pause_play_step = QPushButton('Play Simulation')
        self.button_pause_play_step.clicked.connect(self.upon_pause_play_step_button_clicked)
        self.button_reset = QPushButton('Reset Simulation')
        self.button_reset.clicked.connect(self.upon_reset_button_clicked)
        self.button_redraw_orchard = QPushButton('Redraw Orchard')
        self.button_redraw_orchard.clicked.connect(self.upon_redraw_orchard_button_clicked)
        self.button_reset_all = QPushButton('Reset All')
        self.button_reset_all.clicked.connect(self.upon_reset_all_button_clicked)
        self.button_row1_layout = QHBoxLayout()
        self.button_row1_layout.addWidget(self.button_pause_play_step)
        self.button_row1_layout.addWidget(self.button_reset)
        self.button_row1_layout.addWidget(self.button_redraw_orchard)
        self.button_row1_layout.addWidget(self.button_reset_all)
        self.layout_controls.addLayout(self.button_row1_layout)

        self.button_row2_layout = QHBoxLayout()

        self.button_move_forward = QPushButton('Forward')
        self.button_move_forward.clicked.connect(self.upon_move_forward_button_clicked)
        self.button_move_forward.setToolTip('Move the robot forward, movement amount depends on step size and robot'
                                            'speed. Particle filter will be updated upon movement.')
        self.button_row2_layout.addWidget(self.button_move_forward)

        self.button_move_backward = QPushButton('Backward')
        self.button_move_backward.clicked.connect(self.upon_move_backward_button_clicked)
        self.button_move_backward.setToolTip('Move the robot backward, movement amount depends on step size and robot'
                                             'speed. Particle filter will be updated upon movement.')
        self.button_row2_layout.addWidget(self.button_move_backward)

        self.button_turn_ccw = QPushButton('Turn CCW')
        self.button_turn_ccw.clicked.connect(self.upon_turn_ccw_button_clicked)
        self.button_turn_ccw.setToolTip('Turn the robot counter clockwise, movement amount depends on step size '
                                        'and robot turn speed. Particle filter will be updated upon movement.')
        self.button_row2_layout.addWidget(self.button_turn_ccw)

        self.button_turn_cw = QPushButton('Turn CW')
        self.button_turn_cw.clicked.connect(self.upon_turn_cw_button_clicked)
        self.button_turn_cw.setToolTip('Turn the robot clockwise, movement amount depends on step size '
                                       'and robot turn speed. Particle filter will be updated upon movement.')
        self.button_row2_layout.addWidget(self.button_turn_cw)

        self.button_read_sensor = QPushButton('Read Sensor')
        self.button_read_sensor.clicked.connect(self.upon_read_sensor_button_clicked)
        self.button_read_sensor.setToolTip('Activate the tree location and width (if active) sensors and update the '
                                           'particle filter. Sensor setting can be adjusted below.')
        self.button_row2_layout.addWidget(self.button_read_sensor)

        self.layout_controls.addLayout(self.button_row2_layout)

        self.slider_step_time = HorizontalSliderLayout("Step Time (ms)", self.world.step_time_default,
                                                       self.world.step_time_range)
        self.slider_step_time.slider.setToolTip('Set the time between each step, the movement distance per step will be'
                                                ' affected by this and the robot speed')
        self.slider_step_time.slider.valueChanged.connect(self.upon_slider_step_time_change)
        self.layout_controls.addLayout(self.slider_step_time.layout)

        self.mode = "Continuous Play"
        self.upon_spinbox_mode_choice_value_change()

        self.label_heading_robot_setting = QLabel('Robot Settings')
        self.label_heading_robot_setting.setFont(QFont('Arial', 16))
        self.layout_controls.addWidget(self.label_heading_robot_setting)

        self.slider_speed = HorizontalSliderLayout("Linear Speed (m/s)", self.world.speed_default,
                                                   self.world.speed_range)
        self.slider_speed.slider.setToolTip('Set the speed the robot moves forward and backwards')
        self.slider_speed.slider.valueChanged.connect(self.upon_slider_speed_change)
        self.layout_controls.addLayout(self.slider_speed.layout)

        self.slider_turn_speed = HorizontalSliderLayout("Turn Speed (rad/s)", self.world.turn_speed_default,
                                                        self.world.turn_speed_range)
        self.slider_turn_speed .slider.setToolTip('Set the speed the robot turns')
        self.slider_turn_speed .slider.valueChanged.connect(self.upon_slider_turn_speed_change)
        self.layout_controls.addLayout(self.slider_turn_speed.layout)

        self.label_heading_sensor_setting = QLabel('Sensor Settings')
        self.label_heading_sensor_setting.setFont(QFont('Arial', 16))
        self.layout_controls.addWidget(self.label_heading_sensor_setting)

        self.slider_FOV = HorizontalSliderLayout("Field of View (deg)", self.world.field_of_view_default,
                                                 self.world.field_of_view_range)
        self.slider_FOV.slider.setToolTip('Set the Field of View of the tree location sensor of the robot')
        self.slider_FOV.slider.valueChanged.connect(self.upon_slider_FOV_change)
        self.layout_controls.addLayout(self.slider_FOV.layout)

        self.slider_sensor_range = HorizontalSliderLayout("Sensor Range (m)", self.world.sensor_range_default,
                                                          self.world.sensor_range_range)
        self.slider_sensor_range.slider.setToolTip('Set how far the tree location sensor of the robot can see')
        self.slider_sensor_range.slider.valueChanged.connect(self.upon_slider_sensor_range_change)
        self.layout_controls.addLayout(self.slider_sensor_range.layout)

        self.slider_sensor_accuracy = HorizontalSliderLayout("Sensor Accuracy (m)",
                                                             self.world.location_sensor_accuracy_default,
                                                             self.world.location_sensor_accuracy_range)
        self.slider_sensor_accuracy.slider.setToolTip('Set the standard deviation for the amount of noise added to the'
                                                      ' tree location sensor')
        self.slider_sensor_accuracy.slider.valueChanged.connect(self.upon_slider_sensor_accuracy_change)
        self.layout_controls.addLayout(self.slider_sensor_accuracy.layout)

        self.slider_sample_rate = HorizontalSliderLayout("Sample Rate (Hz)", self.world.sample_rate_default,
                                                         self.world.sample_rate_range, whole_num=True)
        self.slider_sample_rate.slider.setToolTip('Set the number of times per second the tree locations are checked by'
                                                  ' the sensor and the particle filter is updated.')
        self.slider_sample_rate.slider.valueChanged.connect(self.upon_slider_sample_rate_change)
        self.layout_controls.addLayout(self.slider_sample_rate.layout)

        self.slider_movement_accuracy = HorizontalSliderLayout("Movement Accuracy",
                                                               self.world.movement_sensor_accuracy_default,
                                                               self.world.movement_sensor_accuracy_range)
        self.slider_movement_accuracy.slider.setToolTip("Sets the standard deviation of the robot's movement sensor as"
                                                        " a percentage of the movement per step (both angular and "
                                                        "linear)")
        self.slider_movement_accuracy.slider.valueChanged.connect(self.upon_slider_movement_accuracy_change)
        self.layout_controls.addLayout(self.slider_movement_accuracy.layout)

        self.label_heading_PF = QLabel('Particle Filter Settings')
        self.label_heading_PF.setFont(QFont('Arial', 16))
        self.layout_controls.addWidget(self.label_heading_PF)

        self.slider_num_particle = HorizontalSliderLayout("Number of Particles", self.world.robot.num_particle_default,
                                                          self.world.robot.num_particle_range, whole_num=True)
        self.slider_num_particle.slider.setToolTip("Set the number of particles to include in the particle filter")
        self.slider_num_particle.slider.valueChanged.connect(self.upon_slider_num_particle_change)
        self.layout_controls.addLayout(self.slider_num_particle.layout)

        self.slider_particle_movement_noise = HorizontalSliderLayout("Particle Move Noise",
                                                                     self.world.robot.particle_movement_noise_default,
                                                                     self.world.robot.particle_movement_noise_range)
        self.slider_particle_movement_noise.slider.setToolTip("Set the amount of noise to add to each particle's "
                                                              "movement during the particle update phase")
        self.slider_particle_movement_noise.slider.valueChanged.connect(self.upon_slider_particle_movement_noise_change)
        self.layout_controls.addLayout(self.slider_particle_movement_noise.layout)

        self.label_heading_orchard = QLabel('Simulated Orchard Settings')
        self.label_heading_orchard.setFont(QFont('Arial', 16))
        self.layout_controls.addWidget(self.label_heading_orchard)

        self.slider_num_rows = HorizontalSliderLayout("Number of Rows", self.world.num_rows_default,
                                                      self.world.num_rows_range,
                                                      whole_num=True)
        self.slider_num_rows.slider.setToolTip("Set the number of rows to include in the simulated orchard")
        self.slider_num_rows.slider.valueChanged.connect(self.upon_slider_num_rows_change)
        self.layout_controls.addLayout(self.slider_num_rows.layout)

        self.slider_num_columns = HorizontalSliderLayout("Number of Columns", self.world.num_columns_default,
                                                         self.world.num_columns_range, whole_num=True)
        self.slider_num_columns.slider.setToolTip("Set the number of columns to include in the simulated orchard")
        self.slider_num_columns.slider.valueChanged.connect(self.upon_slider_num_columns_change)
        self.layout_controls.addLayout(self.slider_num_columns.layout)

        self.slider_tree_spacing = HorizontalSliderLayout("Tree Spacing (m)", self.world.tree_spacing_default,
                                                          self.world.tree_spacing_range)
        self.slider_tree_spacing.slider.setToolTip("Set the spacing between the trees along the rows in the simulated"
                                                   " orchard")
        self.slider_tree_spacing.slider.valueChanged.connect(self.upon_slider_tree_spacing_change)
        self.layout_controls.addLayout(self.slider_tree_spacing.layout)

        self.slider_row_width = HorizontalSliderLayout("Row Width (m)", self.world.row_width_default,
                                                       self.world.row_width_range)
        self.slider_row_width.slider.setToolTip("Set how far apart the rows of trees are")
        self.slider_row_width.slider.valueChanged.connect(self.upon_slider_row_width_change)
        self.layout_controls.addLayout(self.slider_row_width.layout)

        self.slider_tree_spacing_noise = HorizontalSliderLayout("Tree Spacing Noise",
                                                                self.world.tree_spacing_noise_default,
                                                                self.world.tree_spacing_noise_range)
        self.slider_tree_spacing_noise.slider.setToolTip("Set how accurately the trees are placed according to the "
                                                         "tree spacing set above")
        self.slider_tree_spacing_noise.slider.valueChanged.connect(self.upon_slider_tree_spacing_noise_change)
        self.layout_controls.addLayout(self.slider_tree_spacing_noise.layout)

        self.slider_row_width_noise = HorizontalSliderLayout("Row Width Noise",
                                                             self.world.row_width_noise_default,
                                                             self.world.row_width_noise_range)
        self.slider_row_width_noise.slider.setToolTip("Set how accurately the trees are placed according to row "
                                                      "width set above")
        self.slider_row_width_noise.slider.valueChanged.connect(self.upon_slider_row_width_noise_change)
        self.layout_controls.addLayout(self.slider_row_width_noise.layout)

        # Create matplotlib figure
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)

        plt.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.04)

        self.layout_plots = QVBoxLayout()
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.layout_plots.addWidget(NavigationToolbar(self.canvas, self))
        self.layout_plots.addWidget(self.canvas)

        # Initiate the orchard plot
        self.update_plot(new_orchard=True, new_particles=True)

        self.layout_overall.addLayout(self.layout_controls)
        self.layout_overall.addLayout(self.layout_plots)

        self.window.setLayout(self.layout_overall)
        self.window.show()

    def update_plot(self, new_orchard=False, new_particles=False):
        """
        Update the simulation based on the robots current position and update the plot accordingly
        :param new_orchard: True if redrawing the orchard
        :param new_particles: True if resetting the particles
        """
        # Set the length of the arrow that shows the robot's orientation
        arrow_length = 1.8

        # If redrawing the orchard...
        if new_orchard:
            # Clear both axes
            self.ax1.cla()
            self.ax2.cla()

            # Remake the orchard and path
            self.world.reset()

            # Plot the trees and path
            self.ax1.plot(self.world.xs_trees, self.world.ys_trees, 'b.')
            self.ax1.axis('equal')
            self.ax2.plot(self.world.xs_trees, self.world.ys_trees, 'b.')

            self.ax1.plot(self.world.xs_path, self.world.ys_path, 'k.', markersize=1)
            self.ax2.plot(self.world.xs_path, self.world.ys_path, 'k.', markersize=1)

        else:
            # If not redrawing the orchard, then remove the references to objects that need to move
            self.rect.remove()
            self.arrow.remove()
            self.sensor_FOV_plot.remove()

        # Update the robots position and orientation based on the current step number
        self.world.update_pose()

        # Calculate the position of the rectangle on the upper plot, which shows the viewing area of the lower plot
        self.rect_x_loc = self.world.robot_x - self.rect_x_size / 2
        self.rect_y_loc = self.world.robot_y - self.rect_y_size / 2
        # Create matplotlib patches for the viewing rectangle, the arrow showing the direction, and the wedge showing
        # the sensor field of view
        self.rect = patches.Rectangle((self.rect_x_loc, self.rect_y_loc), self.rect_x_size, self.rect_y_size,
                                      linewidth=1, edgecolor='r', facecolor='none')
        self.arrow = patches.Arrow(self.world.robot_x, self.world.robot_y, np.cos(self.world.robot_angle) *
                                   arrow_length, np.sin(self.world.robot_angle) * arrow_length, width=0.5)

        self.sensor_FOV_plot = patches.Wedge((self.world.robot_x, self.world.robot_y), self.world.sensor_range,
                                             np.degrees(self.world.robot_angle) + 90-self.world.field_of_view/2,
                                             np.degrees(self.world.robot_angle) + 90+self.world.field_of_view/2)

        # If the particles need to be reset...
        if new_particles:
            # If there's already particles on the plot, remove them
            if self.plot1_ref_PF is not None:
                self.plot1_ref_PF.remove()
                self.plot2_ref_PF.remove()
                self.plot2_ref_seen_point.remove()

            # Reset the particle filter
            self.world.robot.reset()

            # Do a sensor reading
            self.world.sensor_reading()

            # Plot the particle filter and the points seen by the sensor and create references for them
            self.plot1_ref_PF = \
                self.ax1.plot(self.world.robot.particles[:, 0], self.world.robot.particles[:, 1], 'r.', markersize=1)[0]
            self.plot2_ref_PF = \
                self.ax2.plot(self.world.robot.particles[:, 0], self.world.robot.particles[:, 1], 'r.', markersize=1)[0]
            self.plot2_ref_seen_point = \
                self.ax2.plot(self.world.x_seen_tree_list_map, self.world.y_seen_tree_list_map, 'y.', markersize=10)[0]

        # If not resetting the particles...
        else:
            # Do a sensor reading
            self.world.sensor_reading()

            # Update the particle filters, note that it uses the sensor reading at the previous position, noise is added
            # to the movement
            move_distance = self.world.speed / self.world.sample_rate
            move_distance += np.random.normal(0, move_distance * self.world.movement_sensor_accuracy)
            angular_distance = self.world.robot_angle - self.world.robot_angle_prev
            angular_distance += np.random.normal(0, abs(angular_distance) * self.world.movement_sensor_accuracy)
            self.world.robot.update_particle_filter(move_distance, angular_distance)

            # Update the plot references with the new particle positions and sensed trees
            self.plot1_ref_PF.set_xdata(self.world.robot.particles[:, 0])
            self.plot1_ref_PF.set_ydata(self.world.robot.particles[:, 1])
            self.plot2_ref_PF.set_xdata(self.world.robot.particles[:, 0])
            self.plot2_ref_PF.set_ydata(self.world.robot.particles[:, 1])
            self.plot2_ref_seen_point.set_xdata(self.world.x_seen_tree_list_map)
            self.plot2_ref_seen_point.set_ydata(self.world.y_seen_tree_list_map)

        # Set the desired viewing area on the lower plot
        self.ax2.set_xlim([self.rect_x_loc, self.rect_x_loc + self.rect_x_size])
        self.ax2.set_ylim([self.rect_y_loc, self.rect_y_loc + self.rect_y_size])
        ratio = self.rect_y_size / self.rect_x_size
        x_left, x_right = self.ax2.get_xlim()
        y_low, y_high = self.ax2.get_ylim()
        self.ax2.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

        # Add the shapes that show the robot's position on the plot
        self.ax1.add_patch(self.rect)
        self.ax2.add_patch(self.arrow)  # , head_width=.3, head_length=.5, fc='k', ec='k', linewidth=2)
        self.ax2.add_patch(self.sensor_FOV_plot)

        # Update the plot
        self.canvas.draw()
        self.canvas.flush_events()

    def move_to_end(self):
        """
        Moves the robot one step at a time until it reaches the end
        """
        # For however many steps left on the path there are...
        for i in range(len(self.world.xs_path)-self.world.step_count-1):
            # Add one to the step count / position along path
            self.world.step_count += 1
            # Update the plot based on the new position
            self.update_plot()

            # Exit loop if simulation is paused
            if self.pause:
                break

    def upon_pause_play_step_button_clicked(self):
        """
        Callback function for when the pause-play / step button is pressed
        """
        # If in pause/play mode...
        if self.spinBox_mode_choice.value() == 0:
            # If paused or not started, then start it
            if self.button_pause_play_step.text() == 'Start':
                self.pause = False
                self.button_pause_play_step.setText('Pause')
                self.move_to_end()

            # Otherwise pause it
            else:
                self.pause = True
                self.button_pause_play_step.setText('Start')
                # time.sleep(.1)

        # If in step mode...
        elif self.spinBox_mode_choice.value() == 1:
            # Move one step and update the plot
            self.world.step_count = self.world.step_count + 1
            self.update_plot()

    def reset_mode(self):
        """
        Pause the sim and set the text to start on the start/pause button
        """
        self.pause = True
        if self.spinBox_mode_choice.value() == 0:
            self.button_pause_play_step.setText('Start')

    def upon_reset_button_clicked(self):
        """
        Callback for reset simulation button, sets the position to the beginning and resets the particle filter
        """
        self.reset_mode()
        self.world.step_count = 0
        self.update_plot(new_particles=True)

    def upon_redraw_orchard_button_clicked(self):
        """
        Callback for redraw orchard button, redraws the orchard based on current settings and resets the simulations
        """
        self.reset_mode()
        self.world.step_count = 0
        self.update_plot(new_orchard=True, new_particles=True)

    def upon_reset_all_button_clicked(self):
        """
        Callback for reset all button, resets all setting to their default, redraws the orchard, and resets the sim
        """
        for slider in HorizontalSliderLayout._slider_registry:
            slider.reset_value()
        self.upon_redraw_orchard_button_clicked()

    def upon_spinbox_mode_choice_value_change(self):
        """
        Callback for changes in the sim mode. Changes the controls based on which mode is chosen
        """
        self.reset_mode()
        self.mode = self.spinBox_mode_choice.textFromValue(self.spinBox_mode_choice.value())
        if self.mode == "Continuous Play":
            self.button_pause_play_step.setDisabled(False)
            self.button_pause_play_step.setText('Start')
            self.button_move_forward.setDisabled(True)
            self.button_move_backward.setDisabled(True)
            self.button_turn_ccw.setDisabled(True)
            self.button_turn_cw.setDisabled(True)
            self.button_read_sensor.setDisabled(True)
            self.slider_step_time.slider.setDisabled(True)

        elif self.mode == "Step Mode":
            # robot.stop
            self.button_pause_play_step.setDisabled(False)
            self.button_pause_play_step.setText('Step')

        elif self.mode == "Control Mode":
            self.button_pause_play_step.setDisabled(True)
            self.button_move_forward.setDisabled(False)
            self.button_move_backward.setDisabled(False)
            self.button_turn_ccw.setDisabled(False)
            self.button_turn_cw.setDisabled(False)
            self.button_read_sensor.setDisabled(False)
            self.slider_step_time.slider.setDisabled(False)

    # Control mode is not programmed so the buttons do nothing currently
    def upon_move_forward_button_clicked(self):
        return

    def upon_move_backward_button_clicked(self):
        return

    def upon_turn_ccw_button_clicked(self):
        return

    def upon_turn_cw_button_clicked(self):
        return

    def upon_read_sensor_button_clicked(self):
        return

    # For all the sliders, when their value changes, update the corresponding variable
    def upon_slider_step_time_change(self):
        self.world.step_time = self.slider_step_time.get_value()
        self.slider_step_time.set_value_to_label()

    def upon_slider_speed_change(self):
        self.world.speed = self.slider_speed.get_value()
        self.slider_speed.set_value_to_label()

    def upon_slider_turn_speed_change(self):
        self.world.turn_speed = self.slider_turn_speed.get_value()
        self.slider_turn_speed.set_value_to_label()

    def upon_slider_FOV_change(self):
        self.world.field_of_view = self.slider_FOV.get_value()
        self.world.robot.field_of_view = self.world.field_of_view
        self.slider_FOV.set_value_to_label()

    def upon_slider_sensor_range_change(self):
        self.world.sensor_range = self.slider_sensor_range.get_value()
        self.world.robot.sensor_range = self.world.sensor_range
        self.slider_sensor_range.set_value_to_label()

    def upon_slider_sensor_accuracy_change(self):
        self.world.location_sensor_accuracy = self.slider_sensor_accuracy.get_value()
        self.slider_sensor_accuracy.set_value_to_label()

    def upon_slider_sample_rate_change(self):
        self.world.sample_rate = self.slider_sample_rate.get_value()
        self.slider_sample_rate.set_value_to_label()

    def upon_slider_movement_accuracy_change(self):
        self.world.movement_sensor_accuracy = self.slider_movement_accuracy.get_value()
        self.slider_movement_accuracy.set_value_to_label()

    def upon_slider_num_particle_change(self):
        self.world.robot.num_particle = self.slider_num_particle.get_value()
        self.slider_num_particle.set_value_to_label()

    def upon_slider_particle_movement_noise_change(self):
        self.world.robot.particle_movement_noise = self.slider_particle_movement_noise.get_value()
        self.slider_particle_movement_noise.set_value_to_label()

    def upon_slider_num_rows_change(self):
        self.world.num_rows = self.slider_num_rows.get_value()
        self.slider_num_rows.set_value_to_label()

    def upon_slider_num_columns_change(self):
        self.world.num_columns = self.slider_num_columns.get_value()
        self.slider_num_columns.set_value_to_label()

    def upon_slider_tree_spacing_change(self):
        self.world.tree_spacing = self.slider_tree_spacing.get_value()
        self.world.robot.tree_spacing = self.world.tree_spacing
        self.slider_tree_spacing.set_value_to_label()

    def upon_slider_row_width_change(self):
        self.world.row_width = self.slider_row_width.get_value()
        self.world.robot.row_width = self.world.row_width
        self.slider_row_width.set_value_to_label()

    def upon_slider_tree_spacing_noise_change(self):
        self.world.tree_spacing_noise = self.slider_tree_spacing_noise.get_value()
        self.slider_tree_spacing_noise.set_value_to_label()

    def upon_slider_row_width_noise_change(self):
        self.world.row_width_noise = self.slider_row_width_noise.get_value()
        self.slider_row_width_noise.set_value_to_label()

class StringBox(QSpinBox):
    """
    Class for a spin box with strings. Found online
    """

    # constructor
    def __init__(self, string_in, parent=None):
        super(StringBox, self).__init__(parent)

        # string values
        strings = string_in

        # calling setStrings method
        self.setStrings(strings)

    # method setString
    # similar to set value method
    def setStrings(self, strings):
        # making strings list
        strings = list(strings)

        # making tuple from the string list
        self._strings = tuple(strings)

        # creating a dictionary
        self._values = dict(zip(strings, range(len(strings))))

        # setting range to it the spin box
        self.setRange(0, len(strings) - 1)

    # overwriting the textFromValue method
    def textFromValue(self, value):
        # returning string from index
        # _string = tuple
        return self._strings[value]


class HorizontalSliderLayout:
    """
    Class for creating each of the sliders along with the corresponding labels and reset button
    """
    # Registry so that all slider objects can be looped through easily
    _slider_registry = []

    def __init__(self, name_of_slider, default, value_range, whole_num=False):
        """
        :param name_of_slider: Name that will be placed to the right of the slider
        :param default: Initial value the slider will be given and value it will reset to
        :param value_range: Range of values that will be possible for the  slider to be set to, default must fall within
        this
        :param whole_num: Whether or not the slider values need to be whole numbers
        """
        # Add this slider to the registry
        self._slider_registry.append(self)

        # Create the label for the slider name and put it on the right side
        self.label = QLabel(name_of_slider)
        self.label.setAlignment(Qt.AlignRight)

        # Create a horizontal slider
        self.slider = QSlider(Qt.Horizontal)

        # Create the label for t he slider value and put it on the left side
        self.label_value = QLabel('0')
        self.label_value.setAlignment(Qt.AlignCenter)

        # Set the default value and value range of the slider
        self.default = default
        self.value_range = value_range
        # Initiate list for the values
        self.values = []
        # Record whether the values should be whole numbers
        self.whole_num = whole_num

        # Configure the slider based on whether it needs whole numbers:
        # If using whole numbers...
        if self.whole_num:
            # The number of positions the slider can have is the number of whole numbers in the given range
            self.num_steps = self.value_range[1] - self.value_range[0]
            # # Record the values the slider can inhabit
            # for i in range(self.num_steps):
            #     self.values.append(self.value_range[0] + i)
            # Set the slider min and max values
            self.slider.setMinimum(self.value_range[0])
            self.slider.setMaximum(self.value_range[1])
            # Set the tick interval for the slider, the ticks are the little dashes above and below the slider
            self.slider.setTickInterval(round((self.value_range[1] - self.value_range[0])/10))

        # If the values don't need to be whole numbers...
        else:
            # Set the number of positions the slider can have to 1000
            self.num_steps = 1000
            # Record what the values each position of the slider correspond to, so the slider value is used as an index
            # to get the actual value from this list

            self.values = self.value_range[0] + (self.value_range[1] - self.value_range[0]) * \
                         (np.arange(0, self.num_steps + 1) / self.num_steps)
            # Set the slider min and max
            self.slider.setMinimum(0)
            self.slider.setMaximum(self.num_steps)
            # Set the tick mark interval
            self.slider.setTickInterval(100)

        # Add tick marks to both sides of the slider
        self.slider.setTickPosition(QSlider.TicksBothSides)

        # Set the initial value to the slider and label
        self.set_value(self.default)
        self.set_value_to_label()

        # Create the reset button and connect its callback
        self.reset_button = QPushButton('r')
        self.reset_button.clicked.connect(self.reset_value)

        # Set the width of each part
        self.label.setFixedWidth(160)
        self.slider.setFixedWidth(300)
        self.label_value.setFixedWidth(55)
        self.reset_button.setFixedWidth(20)

        # Add all the widgets to the layout
        self.layout = QHBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.slider)
        self.layout.addWidget(self.label_value)
        self.layout.addWidget(self.reset_button)


    def slider_pos_from_value(self, value):
        """
        :param value: Value to get the corresponding  slider position for
        :return: slider position that corresponds to the given value
        """
        # Raise value error if the given value is outside the sliders range
        if value < self.value_range[0] or value > self.value_range[1]:
            raise ValueError("Value outside of slider range")

        # Determine the slider position that corresponds to the given value
        if self.whole_num:
            slider_pos = self.slider.value()
        else:
            slider_pos = int(self.num_steps * (value - self.value_range[0]) / (self.value_range[1] -
                                                                               self.value_range[0]))

        return slider_pos

    def set_value_to_label(self):
        """
        Sets the value label to show the value
        """
        # If a whole number, the label corresponds to the slider position
        if self.whole_num:
            self.label_value.setText(str(self.slider.value()))
        # Otherwise, the value is found in the values string with the slider position as an index
        else:
            self.label_value.setText(str(round(self.values[self.slider.value()], 2)))

    def set_value(self, value):
        """
        Sets the slider to a value
        :param value: value to set the slider to
        """
        # Set the position based on the given value
        if self.whole_num:
            self.slider.setValue(value)
        else:
            self.slider.setValue(self.slider_pos_from_value(value))
        # Set the value of the label
        self.set_value_to_label()

    def reset_value(self):
        self.set_value(self.default)

    def get_value(self):
        """
        :return: Returns the current value of the slider
        """
        if self.whole_num:
            return self.slider.value()
        else:
            return self.values[self.slider.value()]


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    app.exec_()

