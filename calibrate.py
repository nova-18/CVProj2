import cv2
import numpy as np
import os
from tqdm import tqdm
import shutil

'''
Takes input video of a checkerboard and saves the calibration data to a file.
The video should be of the (9,6) checkerboard pattern, and the function will extract frames from the video to use for calibration.
Measure the size of the squares in mm and pass it to the function.
The function will save the calibration data (intrinsic matrix and distortion coefficients) to a file in the specified folder.

to load the data, use the following code:

data = np.load(f"{save_folder}/calib_data.npz")
camMatrix = data["camMatrix"]
distCof = data["distCoef"]
'''
def CalibrateCamera(checkerboard_video_path, save_folder, square_size = 16, n_images=40):
        # Create temporary folder to store calibration images
        calib_images_folder = f"{save_folder}/calib_images"
        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(calib_images_folder, exist_ok=True)

        # Debugging variables
        remove_temp = True # set to False to keep the temp folder for debugging
        log = True # set to False to disable logging

        CHESS_BOARD_DIM = (9, 6)

        # The size of Square in the checker board. (in mm)
        SQUARE_SIZE = square_size

        # termination criteria for corner refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        obj_3D = np.zeros((CHESS_BOARD_DIM[0] * CHESS_BOARD_DIM[1], 3), np.float32)

        obj_3D[:, :2] = np.mgrid[0 : CHESS_BOARD_DIM[0], 0 : CHESS_BOARD_DIM[1]].T.reshape(
            -1, 2
        )
        obj_3D *= SQUARE_SIZE

        # Arrays to store object points and image points from all the images.
        obj_points_3D = []  
        img_points_2D = []


        # Read video into calib images
        video = cv2.VideoCapture(checkerboard_video_path)
        frame_count = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            ret1, corners = cv2.findChessboardCorners(frame, CHESS_BOARD_DIM, None)
            if ret1 == True: # use only those images which have checkerboard visible
                cv2.imwrite(f"{calib_images_folder}/frame_{str(frame_count).zfill(4)}.jpg", frame)
                frame_count += 1

        
        files = os.listdir(calib_images_folder)
        n_frames = len(files)
        if n_frames < n_images:
            if log:
                print(f"Not enough frames found in the video. Found {n_frames}, expected {n_images}.")
            return
        
        # uniformly sample n_images from the frames
        step = n_frames // n_images
        actual_files = [files[i] for i in range(0, n_frames, step)][:n_images]


        images_used = 0
        for i, file in tqdm(enumerate(actual_files), total=len(actual_files)):
            imagePath = f"{calib_images_folder}/{file}"

            try:
                image = cv2.imread(imagePath)
                grayScale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            except:
                continue

            ret, corners = cv2.findChessboardCorners(image, CHESS_BOARD_DIM, None)
            if ret == True:
                obj_points_3D.append(obj_3D)
                corners2 = cv2.cornerSubPix(grayScale, corners, (3, 3), (-1, -1), criteria)
                img_points_2D.append(corners2)
                images_used += 1

        if log:
            print(f"Using {images_used} images for calibration")

        if remove_temp:
            shutil.rmtree(calib_images_folder)
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points_3D, img_points_2D, grayScale.shape[::-1], None, None
        )

        if log:
            print("Camera matrix: ", mtx)
            print("Distortion coefficients: ", dist)
            print("Calibration done successfully")

        save_path = f"{save_folder}/calib_data.npz"
        
        if log:
            print("Saving calibration data to: ", save_path)
        
        
        
        # x_pixels = 848 # valo
        # y_pixels = 478 # valo
        # mx = 50 # valo
        # my = mx # valo
        # f = 8 # valo
        # mtx = np.array([
        #     [f*mx, 0, x_pixels/2],
        #     [0, f*my, y_pixels/2],
        #     [0, 0, 1]
        # ], dtype=np.float32) # valo
        
        # dist = np.zeros((5, 1), dtype=np.float32) # valo
        
        np.savez(
            save_path,
            camMatrix=mtx,
            distCoef=dist,
        )

if __name__ == "__main__":
    checkerboard_video_path = "data/input/calib_video.mp4" 
    save_folder = "data/output"  
    square_size = 19
    n_images = 40 

    CalibrateCamera(checkerboard_video_path, save_folder, square_size, n_images)