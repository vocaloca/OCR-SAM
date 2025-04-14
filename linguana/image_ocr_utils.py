from typing import List
import numpy as np
import cv2
from mmocr.utils import poly2bbox
from math import atan2, degrees
import difflib


def group_polygons_in_lines(polygons: List[np.ndarray], 
                            rec_texts: List[str],
                            det_polygon_imgs: List[dict] = None,
                            num_lines: int = None,
                            drop_outliers: bool = True,
                            line_threshold_pct: float = 0.2,
                            angle_threshold: float = 15):
    """Group the polygons into text lines.
    
    This function takes a list of polygons (text bounding boxes) and groups them
    into lines based on their angles and positions. It uses a hierarchical 
    clustering approach that accounts for text rotation and angles.
    
    Args:
        polygons: List of polygons, each as numpy array of points
        rec_texts: List of recognized texts corresponding to polygons
        det_polygon_imgs: List of dictionaries containing angle and original_points
        num_lines: Fixed number of lines to group into (if specified)
        drop_outliers: Whether to remove outlier text boxes from each line
        line_threshold_pct: Percentage of text height/width to use as threshold
        angle_threshold: Maximum angle difference (degrees) to consider polygons
                         aligned in the same line
        
    Returns:
        Tuple of (line_groups, line_polygons) where:
            - line_groups: List of lists containing indices of polygons in each
              line
            - line_polygons: List of polygons representing each line's bounding
              region
    """
    if not polygons or not rec_texts:
        return [], []
    
    # If det_polygon_imgs is not provided, estimate angles from polygons
    if det_polygon_imgs is None:
        det_polygon_imgs = []
        for polygon in polygons:
            # Use axis-aligned bounding box as fallback (0° angle)
            det_polygon_imgs.append({'angle': 0, 'original_points': polygon})
    
    # Calculate features for each polygon: centroid, angle and dimensions
    box_features = []
    for i, polygon in enumerate(polygons):
        # Convert to numpy array and reshape if needed
        polygon_array = np.array(polygon).reshape(-1, 2)
        
        # Calculate centroid
        centroid = polygon_array.mean(axis=0)
        
        # Get angle from det_polygon_imgs if available, or estimate from points
        if 'angle' in det_polygon_imgs[i]:
            angle = det_polygon_imgs[i]['angle']
        else:
            # Estimate text angle (using top edge points)
            # This assumes a specific order of points in the polygon
            # For more robustness, we could use PCA or minimum area rectangle
            if len(polygon_array) >= 4:
                # Use first two points to estimate angle
                dx = polygon_array[1, 0] - polygon_array[0, 0]
                dy = polygon_array[1, 1] - polygon_array[0, 1]
                angle = degrees(atan2(dy, dx))
            else:
                angle = 0  # Fallback
        
        # Calculate bounding box for size information
        bbox = poly2bbox(polygon)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        box_features.append({
            'index': i,
            'polygon': polygon,
            'centroid': centroid,
            'angle': angle,
            'width': width,
            'height': height,
            'text': rec_texts[i] if i < len(rec_texts) else ''
        })
    
    # Define projection function for hierarchical clustering
    def project_point_to_reference_line(point, reference_point, angle):
        """Project a point onto a line with given angle passing through reference"""
        # Convert angle to radians
        angle_rad = np.radians(angle)
        
        # Create direction vector for the line
        direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        
        # Vector from reference point to the point
        point_vector = np.array(point) - np.array(reference_point)
        
        # Project point_vector onto the direction
        projection = np.dot(point_vector, direction) * direction
        
        # Get the projected point
        projected_point = np.array(reference_point) + projection
        
        # Calculate signed distance along the line
        distance_along_line = np.dot(point_vector, direction)
        
        # Calculate perpendicular distance
        perp_distance = np.linalg.norm(point_vector - projection)
        
        return projected_point, distance_along_line, perp_distance
    
    # Initial sort by y-coordinate (for approximate vertical position)
    box_features.sort(key=lambda x: x['centroid'][1])
    
    # Check if we should use the fixed number of lines approach
    if num_lines is not None and num_lines > 0:
        # Hierarchical clustering with adaptive threshold
        clusters = [[box_features[0]]] if box_features else []
        
        for box in box_features[1:]:
            assigned = False
            
            for cluster in clusters:
                # Use average angle and centroid of cluster as reference
                cluster_angle = sum(item['angle'] for item in cluster) / len(cluster)
                ref_centroid = (
                    sum(item['centroid'][0] for item in cluster) / len(cluster),
                    sum(item['centroid'][1] for item in cluster) / len(cluster)
                )
                
                # Check if angle is similar (accounting for rotation errors)
                angle_diff = min(
                    abs(box['angle'] - cluster_angle),
                    abs(box['angle'] - cluster_angle + 360),
                    abs(box['angle'] - cluster_angle - 360)
                )
                
                if angle_diff > angle_threshold:
                    continue
                
                # Get average height for this cluster
                avg_height = sum(item['height'] for item in cluster) / len(cluster)
                # Use height-based threshold (typically lines are separated by 1-2x height)
                threshold = avg_height * line_threshold_pct * 2
                
                # Project this box's centroid onto the cluster's reference line
                _, _, perp_distance = project_point_to_reference_line(
                    box['centroid'], ref_centroid, cluster_angle
                )
                
                # If perpendicular distance is small enough, add to this cluster
                if perp_distance < threshold:
                    cluster.append(box)
                    assigned = True
                    break
            
            if not assigned:
                # Create a new cluster
                clusters.append([box])
        
        # Enforce the expected line count
        if len(clusters) != num_lines:
            # Sort clusters by average y-coordinate
            clusters.sort(
                key=lambda cluster: sum(
                    item['centroid'][1] for item in cluster
                ) / len(cluster)
            )
            
            if len(clusters) > num_lines:
                # Merge clusters until we have the expected count
                while len(clusters) > num_lines:
                    # Find the closest pair of clusters to merge
                    min_distance = float('inf')
                    merge_pair = (0, 0)
                    
                    for i in range(len(clusters) - 1):
                        centroid_i = (
                            sum(item['centroid'][0] for item in clusters[i]) / len(clusters[i]),
                            sum(item['centroid'][1] for item in clusters[i]) / len(clusters[i])
                        )
                        
                        for j in range(i + 1, len(clusters)):
                            centroid_j = (
                                sum(item['centroid'][0] for item in clusters[j]) / len(clusters[j]),
                                sum(item['centroid'][1] for item in clusters[j]) / len(clusters[j])
                            )
                            
                            # Vertical distance
                            dist = abs(centroid_i[1] - centroid_j[1])
                            
                            if dist < min_distance:
                                min_distance = dist
                                merge_pair = (i, j)
                    
                    # Merge the closest pair
                    i, j = merge_pair
                    clusters[i].extend(clusters[j])
                    clusters.pop(j)
            
            elif len(clusters) < num_lines:
                # Split the largest clusters until we have the expected count
                while len(clusters) < num_lines:
                    # Find the largest cluster
                    largest_idx = max(range(len(clusters)), key=lambda i: len(clusters[i]))
                    largest = clusters[largest_idx]
                    
                    if len(largest) < 2:
                        # Can't split a cluster with only one box
                        break
                    
                    # Sort by x-coordinate
                    largest.sort(key=lambda item: item['centroid'][0])
                    
                    # Split into two parts
                    split_point = len(largest) // 2
                    clusters[largest_idx] = largest[:split_point]
                    clusters.insert(largest_idx + 1, largest[split_point:])
        
        # Within each cluster, sort boxes horizontally (along text direction)
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
                
            # Get average angle for this cluster
            cluster_angle = sum(item['angle'] for item in cluster) / len(cluster)
            ref_centroid = (
                sum(item['centroid'][0] for item in cluster) / len(cluster),
                sum(item['centroid'][1] for item in cluster) / len(cluster)
            )
            
            # Calculate projection distances along reading direction
            for item in cluster:
                _, distance, _ = project_point_to_reference_line(
                    item['centroid'], ref_centroid, cluster_angle
                )
                item['projection_distance'] = distance
            
            # Sort by projection distance
            cluster.sort(key=lambda item: item['projection_distance'])
            
            # Remove outliers if requested
            if drop_outliers and len(cluster) > 3:
                is_horizontal = abs(cluster_angle) < 45 or abs(cluster_angle) > 135
                indices = [item['index'] for item in cluster]
                filtered_indices = remove_outliers_from_line(
                    indices, polygons, is_horizontal)
                
                # Update cluster to only contain boxes with indices in filtered_indices
                cluster[:] = [item for item in cluster if item['index'] in filtered_indices]
        
        # Extract line groups and generate line polygons
        line_groups = []
        for cluster in clusters:
            line_groups.append([item['index'] for item in cluster])
        
        # Handle case where we have fewer actual groups than requested num_lines
        while len(line_groups) < num_lines:
            line_groups.append([])
        
        # Generate line bounding polygons
        line_polygons = generate_line_polygons(line_groups, polygons)
        
        return line_groups, line_polygons
    
    # If num_lines is not specified, perform regular clustering without enforcing line count
    # Hierarchical clustering with adaptive threshold
    clusters = [[box_features[0]]] if box_features else []
    
    for box in box_features[1:]:
        assigned = False
        
        for cluster in clusters:
            # Use average angle and centroid of cluster as reference
            cluster_angle = sum(item['angle'] for item in cluster) / len(cluster)
            ref_centroid = (
                sum(item['centroid'][0] for item in cluster) / len(cluster),
                sum(item['centroid'][1] for item in cluster) / len(cluster)
            )
            
            # Check if angle is similar (accounting for rotation errors)
            angle_diff = min(
                abs(box['angle'] - cluster_angle),
                abs(box['angle'] - cluster_angle + 360),
                abs(box['angle'] - cluster_angle - 360)
            )
            
            if angle_diff > angle_threshold:
                continue
            
            # Get average height for this cluster
            avg_height = sum(item['height'] for item in cluster) / len(cluster)
            # Use height-based threshold
            threshold = avg_height * line_threshold_pct * 2
            
            # Project this box's centroid onto the cluster's reference line
            _, _, perp_distance = project_point_to_reference_line(
                box['centroid'], ref_centroid, cluster_angle
            )
            
            # If perpendicular distance is small enough, add to this cluster
            if perp_distance < threshold:
                cluster.append(box)
                assigned = True
                break
        
        if not assigned:
            # Create a new cluster
            clusters.append([box])
    
    # Within each cluster, sort boxes horizontally (along text direction)
    for cluster in clusters:
        if len(cluster) <= 1:
            continue
            
        # Get average angle for this cluster
        cluster_angle = sum(item['angle'] for item in cluster) / len(cluster)
        ref_centroid = (
            sum(item['centroid'][0] for item in cluster) / len(cluster),
            sum(item['centroid'][1] for item in cluster) / len(cluster)
        )
        
        # Calculate projection distances along reading direction
        for item in cluster:
            _, distance, _ = project_point_to_reference_line(
                item['centroid'], ref_centroid, cluster_angle
            )
            item['projection_distance'] = distance
        
        # Sort by projection distance
        cluster.sort(key=lambda item: item['projection_distance'])
        
        # Remove outliers if requested
        if drop_outliers and len(cluster) > 3:
            is_horizontal = abs(cluster_angle) < 45 or abs(cluster_angle) > 135
            indices = [item['index'] for item in cluster]
            filtered_indices = remove_outliers_from_line(
                indices, polygons, is_horizontal)
            
            # Update cluster to only contain boxes with indices in filtered_indices
            cluster[:] = [item for item in cluster if item['index'] in filtered_indices]
    
    # Extract line groups and generate line polygons
    line_groups = []
    for cluster in clusters:
        line_groups.append([item['index'] for item in cluster])
    
    # Generate line bounding polygons
    line_polygons = generate_line_polygons(line_groups, polygons)
    
    return line_groups, line_polygons


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


def match_text_to_lines(line_polygon_rec_texts: List[str], 
                       rec_texts: List[str]) -> List[int]:
    """Match recognized texts from line polygons to original recognized texts
    based on textual similarity.
    
    Args:
        line_polygon_rec_texts: List of texts from line polygons
        rec_texts: List of original recognized texts
        
    Returns:
        List of indices mapping each line_polygon_rec_text to the best 
        matching text in rec_texts
    """
    if not line_polygon_rec_texts or not rec_texts:
        return []
    
    # Create mapping array
    matches = []
    
    # Process texts to improve matching (lowercase, strip whitespace)
    processed_line_texts = [text.lower().strip() for text in line_polygon_rec_texts]
    processed_rec_texts = [text.lower().strip() for text in rec_texts]
    
    # For each line polygon text, find the best matching text
    for line_text in processed_line_texts:
        if not line_text:
            # If line text is empty, append -1 or the first available text
            matches.append(0 if rec_texts else -1)
            continue
            
        # Calculate similarity scores using difflib's SequenceMatcher
        similarity_scores = []
        for rec_text in processed_rec_texts:
            if not rec_text:
                similarity_scores.append(0.0)
                continue
                
            # Calculate string similarity (ratio between 0 and 1)
            similarity = difflib.SequenceMatcher(None, line_text, rec_text).ratio()
            
            # Add word-level matching for better accuracy
            line_words = set(line_text.split())
            rec_words = set(rec_text.split())
            
            # Calculate word overlap (Jaccard similarity)
            if line_words and rec_words:
                word_overlap = len(line_words.intersection(rec_words)) / len(line_words.union(rec_words))
                # Combine character-level and word-level similarity
                similarity = 0.7 * similarity + 0.3 * word_overlap
            
            similarity_scores.append(similarity)
        
        # Find index of highest similarity score
        if similarity_scores:
            best_match = similarity_scores.index(max(similarity_scores))
            matches.append(best_match)
        else:
            matches.append(-1)
    
    return matches


def match_and_sort_texts(line_polygon_rec_texts: List[str], 
                         rec_texts: List[str]) -> List[str]:
    """Match and sort recognized texts based on line polygons.
    
    Args:
        line_polygon_rec_texts: List of texts from line polygons
        rec_texts: List of original recognized texts
        
    Returns:
        List of matched and sorted texts from rec_texts
    """
    # Get matches between line polygon texts and recognized texts
    matches = match_text_to_lines(line_polygon_rec_texts, rec_texts)
    
    # Create sorted result based on matches
    sorted_texts = []
    for match_idx in matches:
        if 0 <= match_idx < len(rec_texts):
            sorted_texts.append(rec_texts[match_idx])
        else:
            # If no good match, use the line polygon text directly
            idx = matches.index(match_idx)
            if idx < len(line_polygon_rec_texts):
                sorted_texts.append(line_polygon_rec_texts[idx])
            else:
                sorted_texts.append("")
    
    return sorted_texts
