import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

root_path = 'C:\\Users\\path_to_dir\\k-means method\\'

# Import the Normalized Data from Excel File
normalized_file_path = f'{root_path}Acri_normalised.xlsx'  # Path to your normalized data file
data = pd.read_excel(normalized_file_path)

# Check if data loaded correctly
print("Data preview:")
print(data.head())

# Extract normalized measurement columns
measurements = data[['Area', 'Mean', 'StdDev',
                     'Circularity', 'Integrated density', 'Aspect ratio']]

# Apply K-Means Clustering on Objects
optimal_k = 5  

# Perform k-means clustering on the normalized measurements
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['cluster'] = kmeans.fit_predict(measurements)

# Save the object-level clustering results (object cluster codes)
object_cluster_file = f'{root_path}object_cluster_codes.xlsx'
data.to_excel(object_cluster_file, index=False)
print(f"Object cluster codes saved to {object_cluster_file}")

# Task 5: Build Image-Level Cluster Proportion Feature Vectors
# Group the data by image name and image class
grouped = data.groupby(['image_name', 'image_class'])

# Function to compute the proportion of objects in each cluster per image
def compute_proportional_features(group, num_clusters):
    # Count the number of objects in each cluster and normalize by the total number of objects in the image
    cluster_counts = group['cluster'].value_counts(normalize=True).reindex(range(num_clusters), fill_value=0)
    return cluster_counts.values  # Return the proportions as a vector

# Apply the function to each group (each image) to get the feature vectors
image_features = grouped.apply(compute_proportional_features, num_clusters=optimal_k)

# Convert the results into a DataFrame
image_features_df = pd.DataFrame(image_features.tolist(), index=image_features.index)

# Reset index so that we have 'image_name' and 'image_class' as columns
image_features_df.reset_index(inplace=True)

# Rename the proportion columns to something meaningful
cluster_columns = [f'cluster_{i}_proportion' for i in range(optimal_k)]
image_features_df.columns = ['image_name', 'image_class'] + cluster_columns

# Save the image-level cluster proportion feature vectors
image_proportion_file = f'{root_path}image_cluster_proportion_vectors.xlsx'
image_features_df.to_excel(image_proportion_file, index=False)
print(f"Image cluster proportion feature vectors saved to {image_proportion_file}")

# Output the first few rows of the final image-level feature vector DataFrame
print(image_features_df.head())
