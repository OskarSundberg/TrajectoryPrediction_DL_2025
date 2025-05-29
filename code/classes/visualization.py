import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from classes.location import Location

class Visualization:
    """
    A class to visualize agent trajectories, counts, time-based data, vectors, and interactions
    based on a location's data. It uses the information from a 'Location' object to create
    various plots and save them to disk.
    """

    def __init__(self, location: Location):
        """
        Initializes the Visualization class with a given location.
        
        Parameters:
        location (Location): An object containing the location data, including image and agent trajectories.
        """
        # Dictionary to map object types to their respective IDs
        self.moving_object_id = {'pedestrian': 0, 'bicyclist': 1, 'light_vehicle': 2, 'heavy_vehicle': 3}
        # Dictionary to map IDs to descriptive names
        self.moving_object_str = {0: 'Pedestrian', 1: 'Bicyclist', 2: 'Light Motor Vehicle', 3: 'Heavy Motor Vehicle'}
        # Location object containing data related to the environment and agents
        self.loc = location
        # Read the image for the location
        self.img = plt.imread(location.img)

    def visualize_all_trajectories(self):
        """
        Visualizes and saves the trajectories of all agents (pedestrians, bicyclists, light and heavy vehicles)
        on the map. Creates subplots for each agent type and saves the plot as an image.
        """
        # Filter the dataframe by types of objects (pedestrians, bicyclists, etc.)
        pedestrians = self.loc.df[self.loc.df.Type == self.moving_object_id['pedestrian']]
        bicyclists = self.loc.df[self.loc.df.Type == self.moving_object_id['bicyclist']]
        lightVeh = self.loc.df[self.loc.df.Type == self.moving_object_id['light_vehicle']]
        heavyVeh = self.loc.df[self.loc.df.Type == self.moving_object_id['heavy_vehicle']]

        # Group by 'ID' for each object type to track individual trajectories
        pedestrian_traj = pedestrians.groupby('ID')
        bicycle_traj = bicyclists.groupby('ID')
        lightVeh_traj = lightVeh.groupby('ID')
        heavyVeh_traj = heavyVeh.groupby('ID')

        # Count the number of distinct agents in each category
        pedestrian_count = pedestrian_traj.ngroups
        bicycle_count = bicycle_traj.ngroups
        light_vehicle_count = lightVeh_traj.ngroups
        heavy_vehicle_count = heavyVeh_traj.ngroups

        # Write these counts to a metadata file
        with open(f'./data/CombinedData/{self.loc.location_name}/agent_info.txt', 'w') as f:
            f.write(f"The number of pedestrian: {pedestrian_count}\n")
            f.write(f"The number of bicycles: {bicycle_count}\n")
            f.write(f"The number of light vehicles: {light_vehicle_count}\n")
            f.write(f"The number of heavy vehicles: {heavy_vehicle_count}\n")

        # Create a subplot for each agent type
        fig, axes = plt.subplots(1, 4, figsize=(15, 5))
        fig.suptitle('All Agent Trajectories', fontsize=14)

        # Set titles and display images for each subplot
        axes[0].set_title('Pedestrian')
        axes[0].imshow(self.img, cmap='gray')

        axes[1].set_title('Light Vehicle')
        axes[1].imshow(self.img, cmap='gray')

        axes[2].set_title('Bike')
        axes[2].imshow(self.img, cmap='gray')

        axes[3].set_title('Heavy Vehicle')
        axes[3].imshow(self.img, cmap='gray')

        # Plot trajectories for each agent type
        for _, track in pedestrian_traj:
            axes[0].plot(track.X, track.Y, '-r', linewidth=0.2)

        for _, track in lightVeh_traj:
            axes[1].plot(track.X, track.Y, '-b', linewidth=0.2)

        for _, track in bicycle_traj:
            axes[2].plot(track.X, track.Y, '-g', linewidth=0.2)

        for _, track in heavyVeh_traj:
            axes[3].plot(track.X, track.Y, '-w', linewidth=0.2)

        # Adjust layout and save the image
        plt.tight_layout()
        plt.savefig(f"./data/CombinedData/{self.loc.location_name}/all_traj.png")

    def visualize_counts(self):
        """
        Visualizes and saves a bar chart of the counts for each type of agent (pedestrian, vehicle, etc.)
        in the location's data. The counts are based on unique 'ID' and 'Type' combinations.
        """
        # Create a new DataFrame with unique combinations of 'ID' and 'Type'
        unique_df = self.loc.df.drop_duplicates(subset=['ID', 'Type'])

        # Count the occurrences of each type in the unique DataFrame
        type_counts = unique_df['Type'].value_counts().reset_index()
        type_counts.columns = ['Type', 'Count']

        # Map type IDs to descriptive labels
        type_counts['Type'] = type_counts['Type'].map(self.moving_object_str)

        # Create and save a bar plot of the type counts
        plt.figure(figsize=(10, 6))
        plt.bar(type_counts['Type'], type_counts['Count'], color='skyblue')
        plt.xlabel('Type')
        plt.ylabel('Count')
        plt.title(f'Counts of Each Type {self.loc.location_name}')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"./data/CombinedData/{self.loc.location_name}/agent_counts.png")

    def visualize_time(self):
        """
        Visualizes and saves a bar chart showing the time spent by each agent type based on the count of 
        appearances, with each appearance treated as a time unit (0.1s).
        """
        # Group by 'Type' and calculate the total count for each type
        total_counts = self.loc.df.groupby('Type').size().reset_index(name='Total Count')

        # Convert total count to time (in seconds)
        total_counts['Time (s)'] = total_counts['Total Count'] * 0.1

        # Map type IDs to descriptive labels
        total_counts['Type'] = total_counts['Type'].map(self.moving_object_str)

        # Create and save a bar plot showing time (in seconds) for each agent type
        plt.figure(figsize=(10, 6))
        plt.bar(total_counts['Type'], total_counts['Time (s)'], color='skyblue')
        plt.xlabel('Type')
        plt.ylabel('Time (s)')
        plt.title(f'Time (s) for Each Type - {self.loc.location_name}')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"./data/CombinedData/{self.loc.location_name}/agent_counts_in_time.png")

    def visualize_vectors(self):
        """
        Visualizes and saves a map showing various points and vectors (e.g., Lane, SideWalk, Crosswalk, etc.)
        that represent the environment's structural features, such as road lanes and walkways.
        """
        df = pd.DataFrame(self.loc.env_vectors)


        # Create a blank image for visualization
        img = cv2.imread(self.loc.img, 1)
        img = cv2.resize(img, (self.loc.img_size_X, self.loc.img_size_y))

        # Define colors for different types of vectors
        colors = {
            'Lane': (255, 0, 0),  # Blue for Lane
            'SideWalk': (0, 255, 0),  # Green for SideWalk
            'PreCrosswalk': (0, 0, 255),  # Red for PreCrosswalk
            'CrossWalk': (255, 255, 0),  # Yellow for CrossWalk
            'Zebra Crossing': (255, 0, 255) # not sure yet
            }

        # Draw points and vectors on the image
        for idx, row in df.iterrows():
            color = colors[row['Type']]

            # Draw points
            cv2.circle(img, row['Top_Left'], 5, color, -1)
            cv2.circle(img, row['Top_Right'], 5, color, -1)
            cv2.circle(img, row['Bottom_Left'], 5, color, -1)
            cv2.circle(img, row['Bottom_Right'], 5, color, -1)

            # Draw directional vectors
            cv2.arrowedLine(img, row['Top_Left'], (row['Top_Left'][0] + row['Top_Left_to_Top_Right'][0], row['Top_Left'][1] + row['Top_Left_to_Top_Right'][1]), color, 2)
            cv2.arrowedLine(img, row['Top_Right'], (row['Top_Right'][0] + row['Top_Right_to_Bottom_Right'][0], row['Top_Right'][1] + row['Top_Right_to_Bottom_Right'][1]), color, 2)
            cv2.arrowedLine(img, row['Bottom_Right'], (row['Bottom_Right'][0] + row['Bottom_Right_to_Bottom_Left'][0], row['Bottom_Right'][1] + row['Bottom_Right_to_Bottom_Left'][1]), color, 2)
            cv2.arrowedLine(img, row['Bottom_Left'], (row['Bottom_Left'][0] + row['Bottom_Left_to_Top_Left'][0], row['Bottom_Left'][1] + row['Bottom_Left_to_Top_Left'][1]), color, 2)

        # Display and save the visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Visualization of Points and Vectors')
        plt.savefig(f"./data/CombinedData/{self.loc.location_name}/env_vectors.png")
        
        
    def plot_agent_interaction_results(
        self,
        src,  # Test source input (tensor: agent positions at each timestamp)
        pred,  # Test prediction from the source (tensor: predicted agent positions)
        tgt,  # Test actual (tensor: ground truth agent positions)
        times_df,  # DataFrame: Contains timestamps for each agent
        type_a,  # Type of agent A to be plotted (e.g., 'pedestrian')
        type_b,  # Type of agent B to be plotted (e.g., 'bicyclist'), can be the same type
        model_name,  # Name of the model used for predictions
        distance_tensor,  # Tensor: Contains pairwise distances between agents
        max_distance,  # Maximum distance between agents to visualize interactions
    ):
        """
        Visualizes agent interactions over time by comparing predicted positions against ground truth positions. 
        Filters interactions based on the distance between agents and types of agents involved.

        Parameters:
        src (tensor): Source agent data (input data).
        pred (tensor): Predicted agent positions.
        tgt (tensor): Ground truth agent positions.
        times_df (DataFrame): DataFrame containing timestamps for each agent.
        type_a (str): Type of agent A (e.g., 'pedestrian').
        type_b (str): Type of agent B (e.g., 'bicyclist').
        model_name (str): The name of the prediction model.
        distance_tensor (tensor): Tensor containing pairwise distances between agents.
        max_distance (float): Maximum distance to consider when plotting interactions.
        """
        # Load the image and resize it to fit the plot
        img = cv2.imread(self.loc.img, 1)
        img = cv2.resize(img, (self.loc.img_size_X, self.loc.img_size_y))

        # Loop through each timestamp in the dataset
        for i in range(len(times_df)):
            time = times_df.iloc[i]['Time']
            matching_indices = times_df[times_df['Time'] == time].index

            # Process only if there are more than one agent at this timestamp
            if len(matching_indices) > 1:
                for j in range(10):
                    # Check if any agents are within the specified max distance
                    result = any(value < max_distance for value in distance_tensor[i, j, 0:self.loc.num_agents])
                    if result:
                        to_continue = True
                        for idx, indice in enumerate(matching_indices):
                            # Skip self-comparison
                            if indice == i:
                                continue
                            # Check if agents are of the correct types and proceed with plotting
                            if src[i, 0, 4] == self.moving_object_id[type_a] and src[indice, 0, 4] == self.moving_object_id[type_b]:
                                to_continue = False
                            else:
                                to_continue = True
                        if to_continue:
                            break

                        # Plot the agent positions and predictions for each matching timestamp
                        plt.figure(figsize=(10, 10))
                        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                        # Plot the source, prediction, and ground truth for each matching index
                        for idx in matching_indices:
                            if idx == i:
                                continue
                            plt.plot(
                                src[idx, :, 0].cpu().numpy(),
                                src[idx, :, 1].cpu().numpy(),
                                color='blue',
                                label='Source',
                                linewidth=1
                            )
                            plt.plot(
                                pred[idx, :, 0].cpu().numpy(),
                                pred[idx, :, 1].cpu().numpy(),
                                color='red',
                                label='Prediction',
                                linewidth=1
                            )
                            plt.plot(
                                tgt[idx, :, 0].cpu().numpy(),
                                tgt[idx, :, 1].cpu().numpy(),
                                color='green',
                                label='Ground Truth',
                                linewidth=1
                            )

                        # Plot the source, prediction, and ground truth for the current agent
                        plt.plot(
                            src[i, :, 0].cpu().numpy(),
                            src[i, :, 1].cpu().numpy(),
                            color='blue',
                            label='Source',
                            linewidth=1
                        )
                        plt.plot(
                            pred[i, :, 0].cpu().numpy(),
                            pred[i, :, 1].cpu().numpy(),
                            color='red',
                            label='Prediction',
                            linewidth=1
                        )
                        plt.plot(
                            tgt[i, :, 0].cpu().numpy(),
                            tgt[i, :, 1].cpu().numpy(),
                            color='green',
                            label='Ground Truth',
                            linewidth=1
                        )

                        # Save the plot and close the figure
                        plt.legend(["Source", "Prediction", "Ground Truth"])
                        plt.title(f'Visualization of Predictions to Ground Truth {model_name} {self.loc.location_name}')
                        plt.savefig(f'./data/Results/{model_name}/{self.loc.location_name}/{type_a}_{type_b}/visualization_{i}.png')
                        plt.close()
                        break

    def plot_range_results(
        self,
        src,  # Test source input (tensor: agent positions at each timestamp)
        pred,  # Test prediction from source (tensor: predicted agent positions)
        tgt,  # Test actual (tensor: ground truth agent positions)
        type,  # Set type if want all use 'ALL'
        start_index,  # Starting index for range of timestamps to visualize
        end_index,  # Ending index for range of timestamps to visualize
        model_name,  # Name of the model used for predictions
    ):
        """
        Visualizes and saves a comparison of predicted and ground truth agent positions within a specified range of timestamps.

        Parameters:
        src (tensor): Source agent data (input data).
        pred (tensor): Predicted agent positions.
        tgt (tensor): Ground truth agent positions.
        type (str): Type of agent to visualize (use 'ALL' for all agent types).
        start_index (int): Starting index of the timestamp range to visualize.
        end_index (int): Ending index of the timestamp range to visualize.
        model_name (str): Name of the prediction model.
        """
        # Load and resize the background image
        img = cv2.imread(self.loc.img, 1)
        img = cv2.resize(img, (self.loc.img_size_X, self.loc.img_size_y))

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Loop over the specified range of timestamps
        for i in range(start_index, end_index):
            plt.plot(
                src[i, :, 0].cpu().numpy(),
                src[i, :, 1].cpu().numpy(),
                color='blue',
                label='Source',
                linewidth=1
            )
            plt.plot(
                pred[i, :, 0].cpu().numpy(),
                pred[i, :, 1].cpu().numpy(),
                color='red',
                label='Prediction',
                linewidth=1
            )
            plt.plot(
                tgt[i, :, 0].cpu().numpy(),
                tgt[i, :, 1].cpu().numpy(),
                color='green',
                label='Ground Truth',
                linewidth=1
            )

        plt.legend(["Source", "Prediction", "Ground Truth"])
        plt.title(f'Visualization of Predictions to Ground Truth {model_name} {self.loc.location_name}')
        plt.savefig(f'/data/Results/{model_name}/{self.loc.location_name}/{start_index}_{end_index}_visualization.png')
        plt.show()

    def plot_prediction_error_normalized_HM(
        self,
        pred,  # Test prediction from source (tensor: predicted agent positions)
        tgt,  # Test actual (tensor: ground truth agent positions)
        model_name,  # Name of the model used for predictions
        location_name,  # Name of the location being visualized
        cmap_type='hot'  # The type of heatmap colormap (default: 'hot', can also use 'viridis')
    ):
        """
        Creates and saves a normalized heatmap visualization of prediction errors between predicted and ground truth positions.

        Parameters:
        pred (tensor): Predicted agent positions.
        tgt (tensor): Ground truth agent positions.
        model_name (str): Name of the prediction model.
        location_name (str): The location name for labeling.
        cmap_type (str): The colormap type for the heatmap ('hot' by default, but 'viridis' is also supported).
        """
        # Load and resize the image background
        img = cv2.imread(self.loc.img, 1)
        img = cv2.resize(img, (self.loc.img_size_X, self.loc.img_size_y))
        img = np.array(img)

        # Reshape the predictions and ground truth to 2D arrays for error computation
        pred_coords = pred.reshape(pred.shape[0] * pred.shape[1], pred.shape[2])
        gt_coords = tgt.reshape(tgt.shape[0] * tgt.shape[1], tgt.shape[2])

        # Calculate the Euclidean distance between predicted and ground truth coordinates
        errors = np.sqrt((pred_coords[:, 0].cpu().numpy() - gt_coords[:, 0].cpu().numpy())**2 + (pred_coords[:, 1].cpu().numpy() - gt_coords[:, 1].cpu().numpy())**2)

        # Set up the heatmap grid using the coordinates of the ground truth
        x_min, x_max = np.min(gt_coords[:, 0].cpu().numpy()), np.max(gt_coords[:, 0].cpu().numpy())
        y_min, y_max = np.min(gt_coords[:, 1].cpu().numpy()), np.max(gt_coords[:, 1].cpu().numpy())

        x_bins = img.shape[1]  # Use the image width as the number of bins
        y_bins = img.shape[0]  # Use the image height as the number of bins

        x_edges = np.linspace(x_min, x_max, x_bins + 1)
        y_edges = np.linspace(y_min, y_max, y_bins + 1)

        # Aggregate errors into heatmap grid cells
        heatmap, x_edges, y_edges = np.histogram2d(
            gt_coords[:, 0].cpu().numpy(),
            gt_coords[:, 1].cpu().numpy(),
            bins=(x_edges, y_edges),
            weights=errors,
            density=False
        )

        # Calculate counts of agents in each grid cell
        counts, _, _ = np.histogram2d(
            gt_coords[:, 0].cpu().numpy(),
            gt_coords[:, 1].cpu().numpy(),
            bins=(x_edges, y_edges)
        )
        
        # Normalize the heatmap by the counts to get a mean error per cell
        heatmap /= counts
        heatmap = np.nan_to_num(heatmap)  # Replace NaNs with zero for empty cells

        # Normalize the heatmap values for display
        norm = Normalize(vmin=np.min(heatmap), vmax=np.max(heatmap))
        heatmap = norm(heatmap)

        # Create the heatmap visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(img, extent=[x_min, x_max, y_min, y_max], aspect='auto')
        plt.imshow(heatmap.T, origin='upper', aspect='auto', extent=[x_min, x_max, y_min, y_max], cmap=cmap_type, alpha=0.6)
        plt.colorbar(label='Normalized Prediction Error')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'Enhanced Heatmap of Prediction Errors {model_name} {self.loc.location_name}')
        plt.savefig(f'/data/Results/{model_name}/{self.loc.location_name}/HeatMap_visualization.png')
        
        

    def plot_prediction_error_enhanced_normalized_HM(
        self,
        pred,  # Test prediction from the model (tensor: predicted agent positions)
        tgt,  # Test actual values (tensor: ground truth agent positions)
        x_bins,  # Number of bins along the x-axis for heatmap (controls granularity)
        y_bins,  # Number of bins along the y-axis for heatmap (controls granularity)
        model_name,  # Name of the model used for predictions
        cmap_type='hot'  # Colormap type for heatmap ('hot' by default, 'viridis' also supported)
    ):
        """
        Generates and visualizes a heatmap of normalized prediction errors between predicted and ground truth positions 
        with enhanced granularity by adjusting the number of bins.

        Parameters:
        pred (tensor): Predicted agent positions from the model.
        tgt (tensor): Ground truth agent positions.
        x_bins (int): Number of bins along the x-axis to control the resolution of the heatmap.
        y_bins (int): Number of bins along the y-axis to control the resolution of the heatmap.
        model_name (str): The name of the prediction model used.
        cmap_type (str): The colormap for the heatmap (default: 'hot', 'viridis' also supported).
        """
        # Load the background image and resize it for consistency in plotting
        img = cv2.imread(self.loc.img, 1)
        img = cv2.resize(img, (self.loc.img_size_X, self.loc.img_size_y))
        img = np.array(img)

        # Reshape the predicted and ground truth coordinates for error calculation
        pred_coords = pred.reshape(pred.shape[0] * pred.shape[1], pred.shape[2])
        gt_coords = tgt.reshape(tgt.shape[0] * tgt.shape[1], tgt.shape[2])

        # Calculate the Euclidean distance (error) between predicted and ground truth positions
        errors = np.sqrt((pred_coords[:, 0].cpu().numpy() - gt_coords[:, 0].cpu().numpy())**2 + 
                         (pred_coords[:, 1].cpu().numpy() - gt_coords[:, 1].cpu().numpy())**2)

        # Define the edges of the grid based on the ground truth coordinates
        x_min, x_max = np.min(gt_coords[:, 0].cpu().numpy()), np.max(gt_coords[:, 0].cpu().numpy())
        y_min, y_max = np.min(gt_coords[:, 1].cpu().numpy()), np.max(gt_coords[:, 1].cpu().numpy())

        # Create grid bins based on the provided x_bins and y_bins parameters
        x_edges = np.linspace(x_min, x_max, x_bins + 1)
        y_edges = np.linspace(y_min, y_max, y_bins + 1)

        # Aggregate the errors into the grid cells and calculate the heatmap
        heatmap, x_edges, y_edges = np.histogram2d(
            gt_coords[:, 0].cpu().numpy(), 
            gt_coords[:, 1].cpu().numpy(), 
            bins=(x_edges, y_edges), 
            weights=errors, 
            density=False
        )

        # Calculate the count of agents per grid cell
        counts, _, _ = np.histogram2d(gt_coords[:, 0].cpu().numpy(), gt_coords[:, 1].cpu().numpy(), bins=(x_edges, y_edges))
        heatmap /= counts  # Normalize the heatmap by dividing by the count of agents
        heatmap = np.nan_to_num(heatmap)  # Replace NaN values with zero for empty cells

        # Normalize the heatmap values for better visualization
        norm = Normalize(vmin=np.min(heatmap), vmax=np.max(heatmap))
        heatmap = norm(heatmap)

        # Plot the heatmap overlaid on the background image
        plt.figure(figsize=(10, 8))
        plt.imshow(img, extent=[x_min, x_max, y_min, y_max], aspect='auto')
        plt.imshow(heatmap.T, origin='upper', aspect='auto', extent=[x_min, x_max, y_min, y_max], cmap=cmap_type, alpha=0.6)
        plt.colorbar(label='Normalized Prediction Error')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'Enhanced Heatmap of Prediction Errors {model_name} {self.loc.location_name}')
        plt.savefig(f'/data/Results/{model_name}/{self.loc.location_name}/HeatMap_visualization.png')

    def plot_prediction_error_HM(
        self,
        pred,  # Test prediction from source (tensor: predicted agent positions)
        tgt,  # Test actual (tensor: ground truth agent positions)
        model_name,  # Name of the model used for predictions
        cmap_type='hot'  # The colormap type for the heatmap ('hot' by default, 'viridis' supported)
    ):
        """
        Generates and visualizes a heatmap of prediction errors by comparing predicted and ground truth agent positions.

        Parameters:
        pred (tensor): Predicted agent positions.
        tgt (tensor): Ground truth agent positions.
        model_name (str): The name of the prediction model used.
        cmap_type (str): The colormap for the heatmap (default: 'hot', 'viridis' supported).
        """
        # Load and resize the background image for visualization
        img = cv2.imread(self.loc.img, 1)
        img = cv2.resize(img, (self.loc.img_size_X, self.loc.img_size_y))
        img = np.array(img)

        # Reshape the predictions and ground truth to 2D arrays for error calculation
        pred_coords = pred.reshape(pred.shape[0] * pred.shape[1], pred.shape[2])
        gt_coords = tgt.reshape(tgt.shape[0] * tgt.shape[1], tgt.shape[2])

        # Calculate the Euclidean distance (error) between predicted and ground truth coordinates
        errors = np.sqrt((pred_coords[:, 0].cpu().numpy() - gt_coords[:, 0].cpu().numpy())**2 + 
                         (pred_coords[:, 1].cpu().numpy() - gt_coords[:, 1].cpu().numpy())**2)

        # Define the edges of the grid based on the ground truth coordinates
        x_min, x_max = np.min(gt_coords[:, 0].cpu().numpy()), np.max(gt_coords[:, 0].cpu().numpy())
        y_min, y_max = np.min(gt_coords[:, 1].cpu().numpy()), np.max(gt_coords[:, 1].cpu().numpy())

        # Use the image dimensions to set the number of bins for the heatmap
        x_bins = img.shape[1]
        y_bins = img.shape[0]

        # Define the bin edges for the heatmap
        x_edges = np.linspace(x_min, x_max, x_bins + 1)
        y_edges = np.linspace(y_min, y_max, y_bins + 1)

        # Aggregate the prediction errors into the grid cells
        heatmap, x_edges, y_edges = np.histogram2d(
            gt_coords[:, 0].cpu().numpy(), 
            gt_coords[:, 1].cpu().numpy(), 
            bins=(x_edges, y_edges), 
            weights=errors, 
            density=False
        )

        # Calculate the counts of agents per grid cell
        counts, _, _ = np.histogram2d(gt_coords[:, 0].cpu().numpy(), gt_coords[:, 1].cpu().numpy(), bins=(x_edges, y_edges))
        heatmap /= counts  # Normalize by dividing by the count of agents in each grid cell
        heatmap = np.nan_to_num(heatmap)  # Replace NaN values (empty cells) with zero

        # Plot the heatmap with the background image
        plt.figure(figsize=(10, 8))
        plt.imshow(img, extent=[x_min, x_max, y_min, y_max], aspect='auto')
        plt.imshow(heatmap.T, origin='upper', aspect='auto', extent=[x_min, x_max, y_min, y_max], cmap=cmap_type, alpha=0.6)
        plt.colorbar(label='Normalized Prediction Error')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'Heatmap of Prediction Errors {model_name} {self.loc.location_name}')
        plt.savefig(f'/data/Results/{model_name}/{self.loc.location_name}/HeatMap_visualization.png')
        
        
    def visualize_roi_torpagatan(
        self,
        pred,  # test prediction from source
        tgt,  # test actual
        model_name,  # model name 
        is_pred=True  # if True, visualize predictions, if False visualize actual ground truth
        ):
        """
        Visualizes the predicted or actual coordinates within the Region of Interest (ROI) for the Torpagatan location.

        This method reads the image, overlays the prediction or actual ground truth points within a specified ROI, 
        and then saves the visualization to the filesystem. The visualization highlights the predicted/actual coordinates 
        and also marks the object of interest in the ROI.

        Parameters:
        pred (tensor): The predicted coordinates for the test set (shape: [num_instances, num_points, 2]).
        tgt (tensor): The ground truth coordinates for the test set (shape: [num_instances, num_points, 2]).
        model_name (str): The name of the model being visualized.
        is_pred (bool): If True, visualizes the predicted points, if False, visualizes the actual ground truth points.

        Returns:
        None: The plot is saved to the specified directory.
        """
        
        # Load and resize the image for the visualization
        img = cv2.imread(self.loc.img, 1)
        img = cv2.resize(img, (self.loc.img_size_X, self.loc.img_size_y))

        # Initialize the plot
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display

        for roi in self.loc.roi:
            # Loop through all instances in the prediction and target data
            for instance in range(pred.shape[0]):
                pred_coords = pred[instance, :, :2].cpu().numpy()  # Get the predicted coordinates
                tgt_coords = tgt[instance, :, :2].cpu().numpy()  # Get the ground truth coordinates

                # Create masks to filter coordinates within the ROI
                roi_mask_pred = (
                    (pred_coords[:, 0] >= roi['roi_x_min']) & (pred_coords[:, 0] <= roi['roi_x_max']) &
                    (pred_coords[:, 1] >= roi['roi_y_min']) & (pred_coords[:, 1] <= roi['roi_y_max'])
                )
                roi_mask_tgt = (
                    (tgt_coords[:, 0] >= roi['roi_x_min']) & (tgt_coords[:, 0] <= roi['roi_x_max']) &
                    (tgt_coords[:, 1] >= roi['roi_y_min']) & (tgt_coords[:, 1] <= roi['roi_y_max'])
                )

                # Apply the masks to get the filtered coordinates
                filtered_pred_coords = pred_coords[roi_mask_pred] if np.any(roi_mask_pred) else np.array([]).reshape(0, 2)
                filtered_tgt_coords = tgt_coords[roi_mask_tgt] if np.any(roi_mask_tgt) else np.array([]).reshape(0, 2)

                # Plot the filtered coordinates (either prediction or ground truth)
                if is_pred:
                    if filtered_pred_coords.size > 0:
                        plt.plot(filtered_pred_coords[:, 0], filtered_pred_coords[:, 1], color='red', linewidth=0.1)
                else:
                    if filtered_tgt_coords.size > 0:
                        plt.plot(filtered_tgt_coords[:, 0], filtered_tgt_coords[:, 1], color='green', linewidth=0.1)

            # Plot the object location within the ROI
            object_x = roi['object_x']
            object_y = roi['object_y']
            object_type = roi['type']
            plt.scatter(object_x, object_y, color='black', s=5, label=object_type)  # Mark the object location
            plt.legend()

            # Save the figure with the appropriate title and file path
            if is_pred:
                plt.title('Visualization of Predictions in ROI')
                plt.savefig(f'./data/Results/{model_name}/{self.loc.location_name}/ROI__prediction{object_type}_{object_x}_{object_y}.png')
            else:
                plt.title('Visualization of Ground Truth in ROI')
                plt.savefig(f'./data/Results/{model_name}/{self.loc.location_name}/ROI_actual_{object_type}_{object_x}_{object_y}.png')


    def visualize_roi_valhallavagen(
            self,
            pred,  # test prediction from source
            tgt,  # test actual
            model_name,  # model name 
            is_pred=True  # if True, visualize predictions, if False visualize actual ground truth
            ):
        """
        Visualizes the predicted or actual coordinates within the Region of Interest (ROI) for the Valhallavagen location.

        This method reads the image, overlays the prediction or actual ground truth points within a specified ROI, 
        and then saves the visualization to the filesystem. The visualization highlights the predicted/actual coordinates 
        and marks specific points in the Valhallavagen location.

        Parameters:
        pred (tensor): The predicted coordinates for the test set (shape: [num_instances, num_points, 2]).
        tgt (tensor): The ground truth coordinates for the test set (shape: [num_instances, num_points, 2]).
        model_name (str): The name of the model being visualized.
        is_pred (bool): If True, visualizes the predicted points, if False, visualizes the actual ground truth points.

        Returns:
        None: The plot is saved to the specified directory.
        """
        
        # Load and resize the image for the visualization
        img = cv2.imread(self.loc.img, 1)
        img = cv2.resize(img, (self.loc.img_size_X, self.loc.img_size_y))

        # Initialize the plot
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display

        # Manually plot points for the Valhallavagen location (hard-coded)
        plt.scatter(325, 583, color='black', s=5)
        plt.scatter(311, 690, color='black', s=5)
        plt.scatter(180, 573, color='black', s=5)
        plt.scatter(166, 674, color='black', s=5)

        # loop over the 2 rois
        for roi in self.loc.roi:
            # Loop through all instances in the prediction and target data
            for instance in range(min(pred.shape[0], 1000)):
                pred_coords = pred[instance, :, :2].cpu().numpy()  # Get the predicted coordinates
                tgt_coords = tgt[instance, :, :2].cpu().numpy()  # Get the ground truth coordinates

                # Create masks to filter coordinates within the ROI
                roi_mask_pred = (
                    (pred_coords[:, 0] >= roi['roi_x_min']) & (pred_coords[:, 0] <= roi['roi_x_max']) &
                    (pred_coords[:, 1] >= roi['roi_y_min']) & (pred_coords[:, 1] <= roi['roi_y_max'])
                )
                roi_mask_tgt = (
                    (tgt_coords[:, 0] >= roi['roi_x_min']) & (tgt_coords[:, 0] <= roi['roi_x_max']) &
                    (tgt_coords[:, 1] >= roi['roi_y_min']) & (tgt_coords[:, 1] <= roi['roi_y_max'])
                )

                # Apply the masks to get the filtered coordinates
                filtered_pred_coords = pred_coords[roi_mask_pred] if np.any(roi_mask_pred) else np.array([]).reshape(0, 2)
                filtered_tgt_coords = tgt_coords[roi_mask_tgt] if np.any(roi_mask_tgt) else np.array([]).reshape(0, 2)

                # Plot the filtered coordinates (either prediction or ground truth)
                if is_pred:
                    if filtered_pred_coords.size > 0:
                        plt.plot(filtered_pred_coords[:, 0], filtered_pred_coords[:, 1], color='red', linewidth=0.1)
                else:
                    if filtered_tgt_coords.size > 0:
                        plt.plot(filtered_tgt_coords[:, 0], filtered_tgt_coords[:, 1], color='green', linewidth=0.1)

            # Get the location name from the ROI for labeling
            location = roi['location']
            # Finalize the plot
            if is_pred:
                plt.title('Visualization of Predictions in ROI')
                plt.savefig(f'./data/Results/{model_name}/{self.loc.location_name}/ROI__prediction_{location}.png')
            else:
                plt.title('Visualization of Ground Truth in ROI')
                plt.savefig(f'./data/Results/{model_name}/{self.loc.location_name}/ROI_actual_{location}.png')