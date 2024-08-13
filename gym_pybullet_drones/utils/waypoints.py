import math
import matplotlib.pyplot as plt
import random
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class WaypointGenerator:
    def __init__(self, radius=3):
        self.radius = radius

    def generate_circle(self, current_x, current_y, current_z, num_waypoints=10):
        angles = np.linspace(0, 2 * np.pi, num_waypoints, endpoint=False)
        x = current_x + self.radius * np.cos(angles)
        y = current_y + self.radius * np.sin(angles)
        z = np.full(num_waypoints, current_z)
        return np.column_stack((x, y, z))

    def generate_figure_8(self, current_x, current_y, current_z, num_waypoints=20):
        angles = np.linspace(0, 2 * np.pi, num_waypoints, endpoint=False)
        x = current_x + self.radius * np.sin(angles) * np.cos(angles)
        y = current_y + self.radius * np.sin(angles)
        z = np.full(num_waypoints, current_z)
        return np.column_stack((x, y, z))

    def generate_square(self, current_x, current_y, current_z, side_length=5):
        half_side = side_length / 2
        waypoints = np.array([
            [current_x - half_side, current_y - half_side, current_z],
            [current_x + half_side, current_y - half_side, current_z],
            [current_x + half_side, current_y + half_side, current_z],
            [current_x - half_side, current_y + half_side, current_z],
            [current_x - half_side, current_y - half_side, current_z]
        ])
        return waypoints

    def generate_oval(self, current_x, current_y, current_z, major_radius=15, minor_radius=10):
        angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        x = current_x + major_radius * np.cos(angles)
        y = current_y + minor_radius * np.sin(angles)
        z = np.full(20, current_z)
        return np.column_stack((x, y, z))

    def plot_waypoints(self, waypoints, title="Drone Trajectory"):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 'bo-', label="Waypoints")
        ax.set_xlabel("X-coordinate")
        ax.set_ylabel("Y-coordinate")
        ax.set_zlabel("Z-coordinate")
        ax.set_title(title)
        plt.grid(True)
        plt.show()

    def generate_random_trajectory(self, current_x, current_y, current_z):
        trajectory_functions = [
            self.generate_circle,
            self.generate_figure_8,
            self.generate_square,
            self.generate_oval
        ]
        chosen_function = random.choice(trajectory_functions)
        
        if chosen_function == self.generate_circle:
            waypoints = chosen_function(current_x, current_y, current_z, num_waypoints=12)
        elif chosen_function == self.generate_figure_8:
            waypoints = chosen_function(current_x, current_y, current_z, num_waypoints=12)
        elif chosen_function == self.generate_square:
            waypoints = chosen_function(current_x, current_y, current_z, side_length=5)
        elif chosen_function == self.generate_oval:
            waypoints = chosen_function(current_x, current_y, current_z)
        
        return waypoints

if __name__ == "__main__":
    waypoint_gen = WaypointGenerator()
    current_x, current_y, current_z = 10, 20, 5

    random_waypoints = waypoint_gen.generate_random_trajectory(current_x, current_y, current_z)
    waypoint_gen.plot_waypoints(random_waypoints, "Random 3D Trajectory")