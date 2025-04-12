import numpy as np
import cv2
import os
import shutil
from cv2 import aruco
from tqdm import tqdm
from render import images_to_video


def estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    Assuming center of aruco marker is at (0,0,0) and the marker is in the XY plane,
    the marker points are defined as follows:
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    
    #print("\n Corners Shape \n", corners.shape)
    corners1 = [corners]
    
    for c in corners1:
        
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    
    rvec = rvecs[0]
    tvec = tvecs[0]
    tvec = np.array(tvec.flatten())

    rotation_matrix, _ = cv2.Rodrigues(rvec)

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = tvec
    
    transformation_matrix = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]) @ transformation_matrix
    #transformation_matrix = np.linalg.inv(transformation_matrix)
    return transformation_matrix, rvec, tvec

def DetectArucoCorners(image):
        marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters() 
        frame = image.copy()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(gray_frame, marker_dict, parameters=parameters)
        if ids is not None:
            return corners, ids
        else:
            return None, None
def look_at_view_matrix(cam_loc: np.ndarray, target_loc: np.ndarray, up: np.ndarray = np.array([0., 0., 1.])) -> np.ndarray:
    """
    Computes the 4x4 view matrix (extrinsic matrix) for a camera looking at a target.

    Parameters:
    - cam_loc (np.ndarray): The camera location (shape: (3,))
    - target_loc (np.ndarray): The point the camera is looking at (shape: (3,))
    - up (np.ndarray): The world up direction (default is [0, 0, 1])

    Returns:
    - view_matrix (np.ndarray): The 4x4 view (extrinsic) matrix
    """
    # Normalize forward vector (camera's -Z axis)
    forward = cam_loc - target_loc
    forward /= np.linalg.norm(forward)

    # Right vector (camera's X axis)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)

    # Recomputed true up vector (camera's Y axis)
    true_up = np.cross(forward, right)

    # Rotation matrix (camera axes as rows)
    R = np.stack([right, true_up, forward], axis=0)

    # Translation vector
    t = -R @ cam_loc

    # Construct the 4x4 view matrix
    view_matrix = np.eye(4)
    view_matrix[:3, :3] = R
    view_matrix[:3, 3] = t

    return view_matrix
def EstimateExtrinsicUsingMarker(input_video_path, intrinsics_path, output_extrinsics_path, temp_result_path=None):
    if temp_result_path is not None:
        os.makedirs(f"{temp_result_path}/tmp1", exist_ok=True)
    
    # Load intrinsic parameters
    intrinsics = np.load(intrinsics_path) # K|0
    
    intrinsic_matrix = intrinsics["camMatrix"]
    distortion_coefficients = intrinsics["distCoef"]

    # Iterate through video
    cap = cv2.VideoCapture(input_video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    extrinsic_matrices = []
    
    for i in tqdm(range(n_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        # Detect ArUco markers and estimate pose
        corners, ids = DetectArucoCorners(frame)
        t = float(i)/float(n_frames) # valo
        test_m = look_at_view_matrix(np.array([-0.6 - 5.0 * t, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])) # valo
        
        if corners is not None:
            extrinsic_matrix, rvec, tvec = estimatePoseSingleMarkers(corners[0][0], 0.165, intrinsic_matrix, distortion_coefficients)
            extrinsic_matrices.append(extrinsic_matrix)
            # print(f"\nExtrinsic Mat {i}: {extrinsic_matrix}\n")
            
            # extrinsic_matrices.append(test_m) # valo
            
            # Optional visualization
            if temp_result_path is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                # Draw normal (Z-axis) vector
                origin = tuple(corners[0][0].mean(axis=0).astype(int))
                end_point, _ = cv2.projectPoints(np.array([[0, 0, 0.165]]), rvec, tvec, intrinsic_matrix, distortion_coefficients)
                end_point = tuple(end_point[0][0].astype(int))
                cv2.arrowedLine(frame, origin, end_point, (255, 0, 0), 3, tipLength=0.2)

                # Save image
                output_path = f"{temp_result_path}/tmp1/output_{str(i+1)}.png"
                cv2.imwrite(output_path, frame)
        else:
            extrinsic_matrices.append(np.eye(4))
            # extrinsic_matrices.append(test_m) # valo
            
    # print("\nExt Mat\n", extrinsic_matrices[50])
    # Save extrinsic matrices
    extrinsic_dict = {str(i+1).zfill(4): matrix for i, matrix in enumerate(extrinsic_matrices)}
    np.savez(output_extrinsics_path, **extrinsic_dict)
    
    if temp_result_path is not None:
        images_to_video(
            f"{temp_result_path}/tmp1",
            f"{temp_result_path}/output.mp4",
            fps=30
        )
        shutil.rmtree(f"{temp_result_path}/tmp1")