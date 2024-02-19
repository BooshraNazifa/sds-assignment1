import numpy as np
from scipy.spatial.distance import cdist

def read_points_from_file(file_path):
    """Read points from the given file path."""
    return np.loadtxt(file_path, delimiter=',')

def closest_points(A, B):
    """Find the closest points in B for each point in A."""
    distances = cdist(A, B, metric='euclidean')
    indices = np.argmin(distances, axis=1)
    return B[indices]

def compute_rigid_transform(A, B):
    """Compute the optimal rigid transformation that aligns A to B."""
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = centroid_B - np.dot(R, centroid_A)
    return R, t

def icp(A, B, max_iterations=200, tolerance=1e-5):
    """Perform the ICP algorithm to compute the rigid transformation from A to B."""
    prev_error = float('inf')
    for i in range(max_iterations):
        B_closest = closest_points(A, B)
        R, t = compute_rigid_transform(A, B_closest)
        A = np.dot(A, R.T) + t
        current_error = np.mean(np.linalg.norm(B_closest - A, axis=1))
        if np.abs(prev_error - current_error) < tolerance:
            break
        prev_error = current_error
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

points_A1 = read_points_from_file("./Examples/PointsA1.txt")
points_B1 = read_points_from_file("./Examples/PointsB1.txt")

print(points_A1)
print(points_B1)
# Compute the transformation matrix using ICP
transformation_matrix = icp(points_A1, points_B1)
print("Transformation Matrix from A to B:")
print(transformation_matrix)