import numpy as np


def smiley_distr(n_head=100, n_mouth=60):
    """
    Generates points that lie exactly on the geometric lines of a 2D smiley face.
    The distribution (random noise) has been removed.

    Args:
        n_head (int): Number of points for the circular head.
        n_eyes (int): (Ignored) The eyes are now 2 single, sharp points.
        n_mouth (int): Number of points for the mouth arc.

    Returns:
        tuple: (all_x_coords, all_y_coords) of the generated points.
    """
    
    # --- 1. Head (Main Circle) ---
    # Radius and center for the main head circle
    R_head = 1.0
    center_x, center_y = 0.0, 0.0
    
    # Generate angles evenly distributed around the circle
    angles_head = np.linspace(0, 2 * np.pi, n_head, endpoint=False)
    
    # Base circle coordinates (points lie exactly on the circumference)
    x_head = center_x + R_head * np.cos(angles_head)
    y_head = center_y + R_head * np.sin(angles_head)
    
    # Removed: Gaussian noise addition for the head distribution
    
    
    # --- 2. Eyes (Two Sharp Points) ---
    # Define centers for the eyes
    eye_1_center = (-2.0/5, 2.0/5)
    eye_2_center = (2.0/5, 2.0/5)
    
    # Generate single, precise points for each eye to represent a sharp dot.
    # The original n_eyes parameter is ignored, resulting in 2 total points for the eyes.
    x_eye_1 = np.array([eye_1_center[0]])
    y_eye_1 = np.array([eye_1_center[1]])

    x_eye_2 = np.array([eye_2_center[0]])
    y_eye_2 = np.array([eye_2_center[1]])
    
    
    # --- 3. Mouth (Arc) ---
    # Radius for the mouth arc
    R_mouth = 3.0/5
    
    # Generate angles for the arc (from 220 degrees to 320 degrees in radians)
    start_angle = np.deg2rad(220)
    end_angle = np.deg2rad(320)
    angles_mouth = np.linspace(start_angle, end_angle, n_mouth)
    
    # Base arc coordinates (points lie exactly on the arc)
    x_mouth = center_x + R_mouth * np.cos(angles_mouth)
    y_mouth = center_y + R_mouth * np.sin(angles_mouth)
    
    # --- 4. Combine all points ---
    all_x = np.concatenate([x_head, x_eye_1, x_eye_2, x_mouth])
    all_y = np.concatenate([y_head, y_eye_1, y_eye_2, y_mouth])
    
    return all_x, all_y



def square_distr(side_length: float = 1.0, points_per_side: int = 30):
    """
    Generates a set of points forming the perimeter of a square centered at (0, 0).

    Args:
        side_length (float): The length of one side of the square. Defaults to 1.0.
        points_per_side (int): The number of points to generate along one side.

    Returns:
        np.ndarray: A NumPy array of shape (N, 2) containing the (x, y) coordinates.
    """
    
    half_side = side_length / 2.0
    
    # Generate points for one side (e.g., from -half_side to half_side)
    coords = np.linspace(-half_side, half_side, points_per_side, endpoint=False)
    
    # --- 1. Top Side (y = half_side) ---
    x_top = coords
    y_top = np.full_like(coords, half_side)
    
    # --- 2. Right Side (x = half_side) ---
    # We use coords[::-1] to go from top to bottom
    x_right = np.full_like(coords, half_side)
    y_right = coords[::-1] 
    
    # --- 3. Bottom Side (y = -half_side) ---
    # We use coords[::-1] to go from right to left
    x_bottom = coords[::-1]
    y_bottom = np.full_like(coords, -half_side)
    
    # --- 4. Left Side (x = -half_side) ---
    # We use coords to go from bottom to top
    x_left = np.full_like(coords, -half_side)
    y_left = coords 
    
    # Combine all points
    all_x = np.concatenate([x_top, x_right, x_bottom, x_left])
    all_y = np.concatenate([y_top, y_right, y_bottom, y_left])
    
    return all_x, all_y