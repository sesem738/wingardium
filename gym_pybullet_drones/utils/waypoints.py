import math
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D

class WaypointGenerator:
    """
    Generates waypoints for different drone trajectories in 3D space.
    """

    def __init__(self, radius=3):
        """
        Initializes the WaypointGenerator with an optional radius.

        Args:
            radius (float): Radius for circle, oval, and figure-8 trajectories. Defaults to 3.
        """
        self.radius = radius

    def generate_circle(self, current_x, current_y, current_z, num_waypoints=10):
        """
        Generates waypoints for a circular trajectory in the XY plane.

        Args:
            current_x (float): X-coordinate of the current position.
            current_y (float): Y-coordinate of the current position.
            current_z (float): Z-coordinate of the current position.
            num_waypoints (int): Number of waypoints to generate. Defaults to 10.

        Returns:
            list: List of waypoints as tuples (x, y, z).
        """
        waypoints = []
        for i in range(num_waypoints):
            angle = (2 * math.pi * i) / num_waypoints
            x = current_x + self.radius * math.cos(angle)
            y = current_y + self.radius * math.sin(angle)
            waypoints.append((x, y, current_z))
        return waypoints

    def generate_figure_8(self, current_x, current_y, current_z, num_waypoints=20):
        """
        Generates waypoints for a figure-8 trajectory in the XY plane.

        Args:
            current_x (float): X-coordinate of the current position.
            current_y (float): Y-coordinate of the current position.
            current_z (float): Z-coordinate of the current position.
            num_waypoints (int): Number of waypoints to generate. Defaults to 20.

        Returns:
            list: List of waypoints as tuples (x, y, z).
        """
        waypoints = []
        for i in range(num_waypoints):
            angle = (2 * math.pi * i) / num_waypoints
            x = current_x + self.radius * math.sin(angle) * math.cos(angle)
            y = current_y + self.radius * math.sin(angle)
            waypoints.append((x, y, current_z))
        return waypoints

    def generate_square(self, current_x, current_y, current_z, side_length=5):
        """
        Generates waypoints for a square trajectory in the XY plane.

        Args:
            current_x (float): X-coordinate of the current position.
            current_y (float): Y-coordinate of the current position.
            current_z (float): Z-coordinate of the current position.
            side_length (float): Length of each side of the square. Defaults to 5.

        Returns:
            list: List of waypoints as tuples (x, y, z).
        """
        half_side = side_length / 2
        waypoints = [
            (current_x - half_side, current_y - half_side, current_z),
            (current_x + half_side, current_y - half_side, current_z),
            (current_x + half_side, current_y + half_side, current_z),
            (current_x - half_side, current_y + half_side, current_z),
            (current_x - half_side, current_y - half_side, current_z),  # Return to starting point
        ]
        return waypoints

    def generate_oval(self, current_x, current_y, current_z, major_radius=15, minor_radius=10):
        """
        Generates waypoints for an oval trajectory in the XY plane.

        Args:
            current_x (float): X-coordinate of the current position.
            current_y (float): Y-coordinate of the current position.
            current_z (float): Z-coordinate of the current position.
            major_radius (float): Length of the major radius. Defaults to 15.
            minor_radius (float): Length of the minor radius. Defaults to 10.

        Returns:
            list: List of waypoints as tuples (x, y, z).
        """
        waypoints = []
        for i in range(20):  # Adjust as needed for desired smoothness
            angle = (2 * math.pi * i) / 20
            x = current_x + major_radius * math.cos(angle)
            y = current_y + minor_radius * math.sin(angle)
            waypoints.append((x, y, current_z))
        return waypoints
    
    def plot_waypoints(self, waypoints, title="Drone Trajectory"):
        """
        Plots the generated waypoints on a 3D graph.

        Args:
            waypoints (list): List of waypoints as tuples (x, y, z).
            title (str): Title for the plot. Defaults to "Drone Trajectory".
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = zip(*waypoints)
        ax.plot(x, y, z, 'bo-', label="Waypoints")
        ax.set_xlabel("X-coordinate")
        ax.set_ylabel("Y-coordinate")
        ax.set_zlabel("Z-coordinate")
        ax.set_title(title)
        plt.grid(True)
        plt.show()

    def generate_random_trajectory(self, current_x, current_y, current_z):
        """
        Generates waypoints for a random trajectory in the XY plane.

        Args:
            current_x (float): X-coordinate of the current position.
            current_y (float): Y-coordinate of the current position.
            current_z (float): Z-coordinate of the current position.

        Returns:
            list: List of waypoints as tuples (x, y, z).
        """
        trajectory_functions = [
            self.generate_circle,
            self.generate_figure_8,
            self.generate_square,
            self.generate_oval
        ]
        chosen_function = random.choice(trajectory_functions)
        
        # Call the randomly chosen function to generate waypoints
        if chosen_function == self.generate_circle:
            waypoints = chosen_function(current_x, current_y, current_z, num_waypoints=12)
        elif chosen_function == self.generate_figure_8:
            waypoints = chosen_function(current_x, current_y, current_z, num_waypoints=12)
        elif chosen_function == self.generate_square:
            waypoints = chosen_function(current_x, current_y, current_z, side_length=5)
        elif chosen_function == self.generate_oval:
            waypoints = chosen_function(current_x, current_y, current_z)  # Use default values for major and minor radii
        
        return waypoints

if __name__ == "__main__":
    # Example usage:
    waypoint_gen = WaypointGenerator()
    current_x = 10
    current_y = 20
    current_z = 5

    # Generate random trajectory
    random_waypoints = waypoint_gen.generate_random_trajectory(current_x, current_y, current_z)
    # random_waypoints = waypoint_gen.generate_figure_8(current_x, current_y, current_z, num_waypoints=20)
    waypoint_gen.plot_waypoints(random_waypoints, "Random 3D Trajectory")