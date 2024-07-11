import math
import matplotlib.pyplot as plt
import random

class WaypointGenerator:
    """
    Generates waypoints for different drone trajectories.
    """

    def __init__(self, radius=3):
        """
        Initializes the WaypointGenerator with an optional radius.

        Args:
            radius (float): Radius for circle, oval, and figure-8 trajectories. Defaults to 10.
        """
        self.radius = radius

    def generate_circle(self, current_x, current_y, num_waypoints=10):
        """
        Generates waypoints for a circular trajectory.

        Args:
            current_x (float): X-coordinate of the current position.
            current_y (float): Y-coordinate of the current position.
            num_waypoints (int): Number of waypoints to generate. Defaults to 10.

        Returns:
            list: List of waypoints as tuples (x, y).
        """
        waypoints = []
        for i in range(num_waypoints):
            angle = (2 * math.pi * i) / num_waypoints
            x = current_x + self.radius * math.cos(angle)
            y = current_y + self.radius * math.sin(angle)
            waypoints.append((x, y))
        return waypoints

    def generate_figure_8(self, current_x, current_y, num_waypoints=20):
        """
        Generates waypoints for a figure-8 trajectory.

        Args:
            current_x (float): X-coordinate of the current position.
            current_y (float): Y-coordinate of the current position.
            num_waypoints (int): Number of waypoints to generate. Defaults to 20.

        Returns:
            list: List of waypoints as tuples (x, y).
        """
        waypoints = []
        for i in range(num_waypoints):
            angle = (2 * math.pi * i) / num_waypoints
            x = current_x + self.radius * math.sin(angle) * math.cos(angle)
            y = current_y + self.radius * math.sin(angle)
            waypoints.append((x, y))
        return waypoints

    def generate_square(self, current_x, current_y, side_length=5):
        """
        Generates waypoints for a square trajectory.

        Args:
            current_x (float): X-coordinate of the current position.
            current_y (float): Y-coordinate of the current position.
            side_length (float): Length of each side of the square. Defaults to 20.

        Returns:
            list: List of waypoints as tuples (x, y).
        """
        half_side = side_length / 2
        waypoints = [
            (current_x - half_side, current_y - half_side),
            (current_x + half_side, current_y - half_side),
            (current_x + half_side, current_y + half_side),
            (current_x - half_side, current_y + half_side),
            (current_x - half_side, current_y - half_side),  # Return to starting point
        ]
        return waypoints

    def generate_oval(self, current_x, current_y, major_radius=15, minor_radius=10):
        """
        Generates waypoints for an oval trajectory.

        Args:
            current_x (float): X-coordinate of the current position.
            current_y (float): Y-coordinate of the current position.
            major_radius (float): Length of the major radius. Defaults to 15.
            minor_radius (float): Length of the minor radius. Defaults to 10.

        Returns:
            list: List of waypoints as tuples (x, y).
        """
        waypoints = []
        for i in range(20):  # Adjust as needed for desired smoothness
            angle = (2 * math.pi * i) / 20
            x = current_x + major_radius * math.cos(angle)
            y = current_y + minor_radius * math.sin(angle)
            waypoints.append((x, y))
        return waypoints
    
    def plot_waypoints(self, waypoints, title="Drone Trajectory"):
        """
        Plots the generated waypoints on a graph.

        Args:
            waypoints (list): List of waypoints as tuples (x, y).
            title (str): Title for the plot. Defaults to "Drone Trajectory".
        """
        x, y = zip(*waypoints)
        plt.plot(x, y, 'bo-', label="Waypoints")
        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")
        plt.title(title)
        plt.grid(True)
        plt.show()

    def generate_random_trajectory(self, current_x, current_y):
        """
        Generates waypoints for a random trajectory.

        Args:
            current_x (float): X-coordinate of the current position.
            current_y (float): Y-coordinate of the current position.

        Returns:
            list: List of waypoints as tuples (x, y).
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
            waypoints = chosen_function(current_x, current_y, num_waypoints=12)
        elif chosen_function == self.generate_figure_8:
            waypoints = chosen_function(current_x, current_y, num_waypoints=12)
        elif chosen_function == self.generate_square:
            waypoints = chosen_function(current_x, current_y, side_length=5)
        elif chosen_function == self.generate_oval:
            waypoints = chosen_function(current_x, current_y)  # Use default values for major and minor radii
        
        return waypoints

# # Example usage:
# waypoint_gen = WaypointGenerator()
# current_x = 10
# current_y = 20

# # Generate random trajectory
# random_waypoints = waypoint_gen.generate_random_trajectory(current_x, current_y)
# waypoint_gen.plot_waypoints(random_waypoints, "Random Trajectory")