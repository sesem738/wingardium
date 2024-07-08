import math
import matplotlib.pyplot as plt

class WaypointGenerator:
    """
    Generates waypoints for different drone trajectories, starting from the current position.
    """

    def __init__(self, current_x, current_y, radius=10):
        """
        Initializes the WaypointGenerator with the current position and optional radius.

        Args:
            current_x (float): X-coordinate of the current position.
            current_y (float): Y-coordinate of the current position.
            radius (float): Radius for circle, oval, and figure-8 trajectories. Defaults to 10.
        """
        self.current_x = current_x
        self.current_y = current_y
        self.radius = radius

    def generate_circle(self, num_waypoints=10):
        """
        Generates waypoints for a circular trajectory, starting from the current position.

        Args:
            num_waypoints (int): Number of waypoints to generate. Defaults to 10.

        Returns:
            list: List of waypoints as tuples (x, y).
        """
        waypoints = []
        for i in range(num_waypoints):
            angle = (2 * math.pi * i) / num_waypoints
            x = self.current_x + self.radius * math.cos(angle)
            y = self.current_y + self.radius * math.sin(angle)
            waypoints.append((x, y))
        return waypoints

    def generate_figure_8(self, num_waypoints=20):
        """
        Generates waypoints for a figure-8 trajectory, starting from the current position.

        Args:
            num_waypoints (int): Number of waypoints to generate. Defaults to 20.

        Returns:
            list: List of waypoints as tuples (x, y).
        """
        waypoints = []
        for i in range(num_waypoints):
            angle = (2 * math.pi * i) / num_waypoints
            x = self.current_x + self.radius * math.sin(angle) * math.cos(angle)
            y = self.current_y + self.radius * math.sin(angle)
            waypoints.append((x, y))
        return waypoints

    def generate_square(self, side_length=20):
        """
        Generates waypoints for a square trajectory, starting from the current position.

        Args:
            side_length (float): Length of each side of the square. Defaults to 20.

        Returns:
            list: List of waypoints as tuples (x, y).
        """
        half_side = side_length / 2
        waypoints = [
            (self.current_x - half_side, self.current_y - half_side),
            (self.current_x + half_side, self.current_y - half_side),
            (self.current_x + half_side, self.current_y + half_side),
            (self.current_x - half_side, self.current_y + half_side),
            (self.current_x - half_side, self.current_y - half_side),  # Return to starting point
        ]
        return waypoints

    def generate_oval(self, major_radius=15, minor_radius=10):
        """
        Generates waypoints for an oval trajectory, starting from the current position.

        Args:
            major_radius (float): Length of the major radius. Defaults to 15.
            minor_radius (float): Length of the minor radius. Defaults to 10.

        Returns:
            list: List of waypoints as tuples (x, y).
        """
        waypoints = []
        for i in range(20):  # Adjust as needed for desired smoothness
            angle = (2 * math.pi * i) / 20
            x = self.current_x + major_radius * math.cos(angle)
            y = self.current_y + minor_radius * math.sin(angle)
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


# # Example usage:
# current_x = 10
# current_y = 20
# waypoint_gen = WaypointGenerator(current_x, current_y)

# # Generate waypoints for a circle
# circle_waypoints = waypoint_gen.generate_circle(12)
# waypoint_gen.plot_waypoints(circle_waypoints, "Circular Trajectory")

# # Generate waypoints for a figure-8
# figure8_waypoints = waypoint_gen.generate_figure_8(25)
# waypoint_gen.plot_waypoints(figure8_waypoints, "Figure-8 Trajectory")