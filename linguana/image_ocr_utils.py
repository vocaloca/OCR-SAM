from typing import List
import numpy as np
import cv2
from mmocr.utils import poly2bbox


def group_polygons_in_lines(polygons: List[np.ndarray], 
                            rec_texts: List[str],
                            det_polygon_imgs: List[dict] = None,
                            num_lines: int = None,
                            drop_outliers: bool = True):
    """Group the polygons into text lines.
    
    This function takes a list of polygons (text bounding boxes) and groups them
    into lines based on their angles and positions. It uses a two-step approach:
    1. Group polygons by similar rotation angles
    2. For each angle group, cluster the polygons into lines
    
    Args:
        polygons: List of polygons, each as numpy array of points
        rec_texts: List of recognized texts corresponding to polygons
        det_polygon_imgs: List of dictionaries containing angle and original_points information
        num_lines: Fixed number of lines to group into (if specified)
        drop_outliers: Whether to remove outlier text boxes from each line
        
    Returns:
        Tuple of (line_groups, line_polygons) where:
            - line_groups: List of lists containing indices of polygons in each line
            - line_polygons: List of polygons representing each line's bounding region
    """
    if not polygons or not rec_texts:
        return [], []
    
    # If det_polygon_imgs is not provided, estimate angles from polygons
    if det_polygon_imgs is None:
        det_polygon_imgs = []
        for polygon in polygons:
            # Use axis-aligned bounding box as fallback (0° angle)
            det_polygon_imgs.append({'angle': 0, 'original_points': polygon})
        
    # Extract exact number of lines from rec_texts if num_lines is provided
    if num_lines is not None and num_lines > 0:
        # Get angle information from the rotated crops
        angles = [det_polygon_img['angle'] for det_polygon_img in det_polygon_imgs]
        
        # Get centroids for all polygons
        centroids = []
        for i, polygon in enumerate(polygons):
            polygon_array = np.array(polygon).reshape(-1, 2)
            centroid = polygon_array.mean(axis=0)
            centroids.append((centroid, i))
        
        # Determine if text is primarily horizontal or vertical
        # Check most common angle
        primary_angles = [abs(angle) < 45 or abs(angle) > 135 for angle in angles]
        is_horizontal = sum(primary_angles) > len(primary_angles) / 2
        
        # Sort by y-coordinate (for horizontal) or x-coordinate (for vertical)
        sort_idx = 1 if is_horizontal else 0
        
        # Sort all indices by vertical position first
        sorted_centroids = sorted(centroids, key=lambda x: x[0][sort_idx])
        sorted_indices = [c[1] for c in sorted_centroids]
        
        # Divide into exactly num_lines groups
        lines = []
        indices_per_line = max(1, len(sorted_indices) // num_lines)
        
        for i in range(num_lines):
            start_idx = i * indices_per_line
            # For the last line, include all remaining indices
            end_idx = (i + 1) * indices_per_line if i < num_lines - 1 else len(sorted_indices)
            
            if start_idx < len(sorted_indices):
                # Get indices for this line
                line_indices = sorted_indices[start_idx:end_idx]
                
                # Sort horizontally within the line
                other_sort_idx = 0 if is_horizontal else 1
                line_centroids = [(centroids[j][0][other_sort_idx], j) for j in line_indices]
                line_indices = [c[1] for c in sorted(line_centroids, key=lambda x: x[0])]
                
                # Remove outliers if requested and we have enough boxes
                if drop_outliers and len(line_indices) > 3:
                    line_indices = remove_outliers_from_line(
                        line_indices, polygons, is_horizontal)
                
                lines.append(line_indices)
        
        # Handle case where we have fewer actual groups than requested num_lines
        while len(lines) < num_lines:
            lines.append([])
            
        # Generate line bounding polygons
        line_polygons = generate_line_polygons(lines, polygons)
            
        return lines, line_polygons
    
    # If num_lines is not specified, fall back to clustering approach
    # Get angle information from the rotated crops 
    angles = [det_polygon_img['angle'] for det_polygon_img in det_polygon_imgs]
    
    # Step 1: Group polygons by similar angles (with tolerance)
    angle_tolerance = 5  # degrees
    angle_groups = {}
    
    for i, angle in enumerate(angles):
        # Find a matching angle group or create a new one
        matched = False
        for group_angle in angle_groups:
            if abs(angle - group_angle) < angle_tolerance:
                angle_groups[group_angle].append(i)
                matched = True
                break
        
        if not matched:
            angle_groups[angle] = [i]
            
    # Step 2: For each angle group, sort polygons into lines based on position
    lines = []
    for angle, indices in angle_groups.items():
        if len(indices) <= 1:
            # Single polygon in this angle group, it's its own line
            lines.append(indices)
            continue
            
        # Get the centroids of polygons in this angle group
        centroids = []
        for idx in indices:
            polygon = np.array(polygons[idx]).reshape(-1, 2)
            centroid = polygon.mean(axis=0)
            centroids.append(centroid)
        
        centroids = np.array(centroids)
        
        # For this angle, determine primary direction (horizontal or vertical)
        is_horizontal = abs(angle) < 45 or abs(angle) > 135
        
        # Sort by y-coordinate (horizontal) or x-coordinate (vertical text)
        sort_idx = 1 if is_horizontal else 0
        
        # Group into lines based on position
        # Calculate a reasonable line spacing threshold
        avg_height = 0
        for idx in indices:
            polygon = polygons[idx]
            bbox = poly2bbox(polygon)
            height = max(bbox[3] - bbox[1], bbox[2] - bbox[0]) / 2
            avg_height += height
            
        avg_height /= len(indices)
        line_threshold = avg_height * 0.7  # Adjust this threshold as needed
        
        # Sort indices by the primary coordinate
        sorted_indices = [x for _, x in sorted(
            zip(centroids[:, sort_idx], indices))]
        
        # Group into lines based on position
        current_line = [sorted_indices[0]]
        current_coord = centroids[indices.index(sorted_indices[0])][sort_idx]
        
        line_groups = []
        
        for i in range(1, len(sorted_indices)):
            idx = sorted_indices[i]
            idx_in_centroids = indices.index(idx)
            coord = centroids[idx_in_centroids][sort_idx]
            
            # If polygon is close enough to current line, add it
            if abs(coord - current_coord) < line_threshold:
                current_line.append(idx)
            else:
                # Remove outliers if requested and we have enough boxes
                if drop_outliers and len(current_line) > 3:
                    current_line = remove_outliers_from_line(
                        current_line, polygons, is_horizontal)
                    
                # Sort by the other coordinate
                other_sort_idx = 0 if is_horizontal else 1
                
                # Create a list of coordinates for sorting
                other_coords = []
                for line_idx in current_line:
                    line_idx_in_centroids = indices.index(line_idx)
                    other_coord = centroids[line_idx_in_centroids][other_sort_idx]
                    other_coords.append(other_coord)
                
                # Sort the current line by the other coordinate
                current_line = [x for _, x in sorted(
                    zip(other_coords, current_line))]
                
                # Start a new line
                line_groups.append(current_line)
                current_line = [idx]
                current_coord = coord
        
        # Don't forget to add the last line
        if current_line:
            # Remove outliers if requested and we have enough boxes
            if drop_outliers and len(current_line) > 3:
                current_line = remove_outliers_from_line(
                    current_line, polygons, is_horizontal)
            
            # Sort the line by the other coordinate
            other_sort_idx = 0 if is_horizontal else 1
            
            # Create a list of coordinates for sorting
            other_coords = []
            for line_idx in current_line:
                line_idx_in_centroids = indices.index(line_idx)
                other_coord = centroids[line_idx_in_centroids][other_sort_idx]
                other_coords.append(other_coord)
            
            # Sort the current line by the other coordinate
            current_line = [x for _, x in sorted(
                zip(other_coords, current_line))]
            
            line_groups.append(current_line)
        
        lines.extend(line_groups)
    
    # Generate line bounding polygons
    line_polygons = generate_line_polygons(lines, polygons)
        
    return lines, line_polygons

def remove_outliers_from_line(line_indices, polygons, is_horizontal):
    """Remove outlier text boxes from a line based on position
    
    Args:
        line_indices: List of indices for boxes in this line
        polygons: List of all polygons
        is_horizontal: Whether text is primarily horizontal
        
    Returns:
        Filtered list of indices with outliers removed
    """
    if len(line_indices) <= 3:  # Need enough data points
        return line_indices
        
    # Get centroids for all polygons in the line
    centroids = []
    for idx in line_indices:
        polygon = np.array(polygons[idx]).reshape(-1, 2)
        centroid = polygon.mean(axis=0)
        centroids.append(centroid)
        
    # Choose coordinate based on text orientation
    # For horizontal text, we check vertical (y) outliers
    # For vertical text, we check horizontal (x) outliers
    outlier_coord_idx = 1 if is_horizontal else 0
    
    # Get coordinates for outlier detection
    coords = [c[outlier_coord_idx] for c in centroids]
    
    # Calculate quartiles and IQR
    q1 = np.percentile(coords, 25)
    q3 = np.percentile(coords, 75)
    iqr = q3 - q1
    
    # Define bounds for outliers (standard 1.5 * IQR rule)
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Filter out outliers
    filtered_indices = []
    for i, coord in enumerate(coords):
        if lower_bound <= coord <= upper_bound:
            filtered_indices.append(line_indices[i])
            
    return filtered_indices if filtered_indices else line_indices  # Fallback
    
def generate_line_polygons(lines, polygons):
    """Generate a bounding polygon for each line of text
    
    Args:
        lines: List of lists containing polygon indices for each line
        polygons: List of all polygons
        
    Returns:
        List of polygons, each with 4 points (8 values) representing
        the bounding region of each line
    """
    line_polygons = []
    
    for line in lines:
        if not line:  # Skip empty lines
            line_polygons.append(np.zeros((4, 2), dtype=np.float32).flatten())
            continue
            
        # Collect all points from polygons in this line
        points = []
        for idx in line:
            polygon = np.array(polygons[idx]).reshape(-1, 2)
            points.extend(polygon.tolist())
            
        points = np.array(points)
        
        # Check if we have enough points
        if len(points) < 4:
            # Fallback: create a simple bounding box
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            box = np.array([
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max]
            ])
        else:
            # Find the convex hull of all polygon points
            hull = cv2.convexHull(points.astype(np.float32))
            
            # Simplify to a rotated rectangle (4 points)
            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect)
        
        # Return as flattened array [x1, y1, x2, y2, x3, y3, x4, y4]
        line_polygons.append(box.flatten())
        
    return line_polygons
