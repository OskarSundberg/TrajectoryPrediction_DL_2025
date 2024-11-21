import math
import torch
import numpy as np
import pandas as pd

class Distances:
    """
    A class for calculating various types of distances, including Euclidean distance, 
    distances from points to lines, and distances between vectors.
    """

    def calculate_distances(self, point, points_list):
        """
        Calculate the Euclidean distance from a given point to all other points in a list.

        Parameters:
        point (Tensor): A tensor representing the point [X, Y].
        points_list (list of lists): A list of lists, where each inner list represents a point [X, Y].

        Returns:
        Tensor: A tensor of Euclidean distances from the given point to each point in points_list.
        """
        point = point.unsqueeze(0)  # Convert to 2D tensor for broadcasting
        points_tensor = torch.tensor(points_list, dtype=torch.float32)
        distances = torch.sqrt(torch.sum((points_tensor - point) ** 2, dim=1))
        return distances

    def distance_to_line(self, x, y, point1, point2):
        """
        Calculate the perpendicular distance from the point (x, y) to the line defined by two points (point1 and point2).

        Parameters:
        x, y (float): Coordinates of the point.
        point1, point2 (tuple): Coordinates of the two points defining the line.

        Returns:
        float: Perpendicular distance from the point to the line.
        """
        x1, y1 = point1
        x2, y2 = point2
        
        # Numerator of the perpendicular distance formula
        numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
        
        # Denominator of the perpendicular distance formula
        denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        
        # Perpendicular distance from the point to the line
        distance = numerator / denominator
        
        return distance

    def calculate_distance_to_nearest_vector(self, x, y, index, df):
        """
        Calculate the perpendicular distance from a point (x, y) to the nearest vector 
        in a polygon defined by four corners, using a dataframe with the coordinates.

        Parameters:
        x, y (float): Coordinates of the point.
        index (int): Index of the row in the dataframe.
        df (DataFrame): DataFrame containing polygon vertices.

        Returns:
        float: Minimum perpendicular distance to one of the polygon's sides.
        """
        row = df.iloc[index]
        
        # Extract coordinates from the row
        top_left = row['Top_Left']
        top_right = row['Top_Right']
        bottom_left = row['Bottom_Left']
        bottom_right = row['Bottom_Right']
        
        # Compute the distances to each of the four vectors (edges of the polygon)
        distances = [
            self.distance_to_line(x, y, top_left, top_right),
            self.distance_to_line(x, y, top_right, bottom_right),
            self.distance_to_line(x, y, bottom_right, bottom_left),
            self.distance_to_line(x, y, bottom_left, top_left)
        ]
        
        # Return the minimum distance
        return min(distances)

    def euclidean_distance(self, x1, y1, x2, y2):
        """
        Calculate the Euclidean distance between two points (x1, y1) and (x2, y2).

        Parameters:
        x1, y1, x2, y2 (float): Coordinates of the two points.

        Returns:
        float: Euclidean distance between the two points.
        """
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def check_coordinate_in_squares(self, x, y, index, df):
        """
        Check if a point (x, y) lies within a square defined by four corners in the dataframe.

        Parameters:
        x, y (float): Coordinates of the point.
        index (int): Index of the row in the dataframe.
        df (DataFrame): DataFrame containing square coordinates.

        Returns:
        int: 0 if the point is inside the square, None if outside.
        """
        row = df.iloc[index]
        # Extract coordinates from the row
        top_left = row['Top_Left']
        top_right = row['Top_Right']
        bottom_left = row['Bottom_Left']
        bottom_right = row['Bottom_Right']
        
        # Compute the coordinates of the square's bounding box
        min_x = min(top_left[0], top_right[0], bottom_left[0], bottom_right[0])
        max_x = max(top_left[0], top_right[0], bottom_left[0], bottom_right[0])
        min_y = min(top_left[1], top_right[1], bottom_left[1], bottom_right[1])
        max_y = max(top_left[1], top_right[1], bottom_left[1], bottom_right[1])

        # Check if the point is within the square
        if min_x <= x <= max_x and min_y <= y <= max_y:
            return 0

        return None

    def calculate_width_to_nearest_vector(self, intersection_x, intersection_y, df):
        """
        Calculate the perpendicular distance from an intersection point (intersection_x, intersection_y) 
        to the nearest vector (edge of a square defined in the dataframe).

        Parameters:
        intersection_x, intersection_y (float): Coordinates of the intersection point.
        df (DataFrame): DataFrame containing square coordinates.

        Returns:
        float: Minimum distance from the intersection point to one of the vectors.
        """
        distances = []
        for index, row in df.iterrows():
            top_left_x, top_left_y = row['Top_Left']
            top_right_x, top_right_y = row['Top_Right']
            bottom_left_x, bottom_left_y = row['Bottom_Left']
            bottom_right_x, bottom_right_y = row['Bottom_Right']
            
            # Compute distance from the intersection point to each side of the square
            distance = self.point_to_line_distance(intersection_x, intersection_y, top_left_x, top_left_y, top_right_x, top_right_y)
            distance = min(distance, self.point_to_line_distance(intersection_x, intersection_y, top_right_x, top_right_y, bottom_right_x, bottom_right_y))
            distance = min(distance, self.point_to_line_distance(intersection_x, intersection_y, bottom_right_x, bottom_right_y, bottom_left_x, bottom_left_y))
            distance = min(distance, self.point_to_line_distance(intersection_x, intersection_y, bottom_left_x, bottom_left_y, top_left_x, top_left_y))
            distances.append(distance)
        
        min_distance = min(distances)
        return min_distance

    def compute_euclidean_distance_objects(self, positions):
        """
        Compute the Euclidean distance between multiple agents over a sequence of positions.

        Parameters:
        positions (Tensor): A tensor of shape (num_agents, sequence_length, 2) representing agent positions.

        Returns:
        Tensor: A tensor of distances between all pairs of agents at each time step.
        """
        num_agents, sequence_length, _ = positions.shape
        
        # Initialize distances tensor
        distances = torch.zeros(num_agents, sequence_length, num_agents)
        
        # Compute pairwise distances
        for i in range(num_agents):
            for j in range(sequence_length):
                for k in range(num_agents):
                    # Get X and Y coordinates for agents i and k at time step j
                    agent_i_coords = positions[i, j, :]
                    agent_k_coords = positions[k, j, :]

                    # Skip the distance calculation if any coordinate is zero
                    if torch.any(agent_i_coords == 0) or torch.any(agent_k_coords == 0):
                        distances[i, j, k] = -1
                    else:
                        dist = torch.norm(agent_i_coords - agent_k_coords)
                        distances[i, j, k] = dist

        # Set distances along the diagonal to 0 (distance to self)
        for i in range(num_agents):
            distances[i, :, i] = 0
        
        return distances

    def point_to_line_distance(self, px, py, x1, y1, x2, y2):
        """
        Calculate the perpendicular distance from a point (px, py) to a line segment defined by two points (x1, y1) and (x2, y2).

        Parameters:
        px, py (float): Coordinates of the point.
        x1, y1, x2, y2 (float): Coordinates of the two points defining the line segment.

        Returns:
        float: Perpendicular distance from the point to the line segment.
        """
        dx = x2 - x1
        dy = y2 - y1
        if dx == dy == 0:  # The line segment is actually a point
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        t = ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)
        t = max(0, min(1, t))
        nearest_x = x1 + t * dx
        nearest_y = y1 + t * dy
        distance = np.sqrt((px - nearest_x)**2 + (py - nearest_y)**2)
        return distance

    def calculate_intersection(self, row):
        """
        Calculate the intersection point of the top-left and bottom-right corners of a square, 
        assuming a square structure defined in the dataframe.

        Parameters:
        row (Series): A row from the dataframe containing square coordinates.

        Returns:
        tuple: Coordinates of the intersection point, or None if data is missing.
        """
        if row.isnull().any():
            return None
        else:
            x_intersection = (row['Top_Left'][0] + row['Bottom_Right'][0]) / 2
            y_intersection = (row['Top_Left'][1] + row['Bottom_Right'][1]) / 2
            return x_intersection, y_intersection

    def calculate_vectors(self, data):
        """
        Calculate vectors for the edges of squares from a given set of data.

        Parameters:
        data (list of dict): List of dictionaries with square corner coordinates.

        Returns:
        DataFrame: DataFrame containing vectors for each square.
        """
        df = pd.DataFrame(data)
        # Calculate vectors for square edges
        df['Top_Left_to_Top_Right'] = df.apply(lambda row: (row['Top_Right'][0] - row['Top_Left'][0], row['Top_Right'][1] - row['Top_Left'][1]), axis=1)
        df['Top_Right_to_Bottom_Right'] = df.apply(lambda row: (row['Bottom_Right'][0] - row['Top_Right'][0], row['Bottom_Right'][1] - row['Top_Right'][1]), axis=1)
        df['Bottom_Right_to_Bottom_Left'] = df.apply(lambda row: (row['Bottom_Left'][0] - row['Bottom_Right'][0], row['Bottom_Left'][1] - row['Bottom_Right'][1]), axis=1)
        df['Bottom_Left_to_Top_Left'] = df.apply(lambda row: (row['Top_Left'][0] - row['Bottom_Left'][0], row['Top_Left'][1] - row['Bottom_Left'][1]), axis=1)
        
        return df
