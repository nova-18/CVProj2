import glfw
from OpenGL.GL import *
import numpy as np
import cv2
import shutil
import os
from tqdm import tqdm
from utils.graphics import Object, Camera, Shader, FBO, object_shader, GetObjProps

def InitializeGL(height, width):
    # Initialize GLFW
    if not glfw.init():
        raise Exception("GLFW can't be initialized")

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)  # Headless / invisible window
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)  # For MacOS compatibility
    
    window = glfw.create_window(width, height, "Offscreen", None, None)
    if not window:
        glfw.terminate()
        raise Exception("GLFW window can't be created")

    glfw.make_context_current(window)

    print("Renderer:", glGetString(GL_RENDERER).decode())
    print("Vendor:  ", glGetString(GL_VENDOR).decode())
    print("Version: ", glGetString(GL_VERSION).decode())

    glViewport(0, 0, width, height)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    return window

def images_to_video(input_folder, output_video_path, fps=30):
    first_frame = cv2.imread(f"{input_folder}/output_1.png")
    if first_frame is None:
        raise ValueError("Could not read the first image.")

    height, width, _ = first_frame.shape
    size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, size)

    for i in range(len(os.listdir(input_folder))):
        frame_path = f"{input_folder}/output_{i+1}.png"
        img = cv2.imread(frame_path)
        if img is None:
            print(f"Warning: Skipping unreadable image: {frame_path}")
            continue
        out.write(img)

    out.release()
    print(f"Video saved to: {output_video_path}")

'''
Takes input video path, input depth video path, output video path, extrinsic matrices path, intrinsics path, and object file path.
Renders video of virtual object, places it in real footage according to depth video, and saves it to output video path.
Uses OpenGL for rendering and OpenCV for video processing.
'''
def Render(input_video_path, input_depth_video_path, output_path, extrinsic_matrices_path, intrinsics_path, obj_file_path):
    os.makedirs(f"{output_path}/tmp", exist_ok=True)

    cap = cv2.VideoCapture(input_video_path)
    ret, frame = cap.read()
    height, width, _ = frame.shape

    window = InitializeGL(height, width)
    fbo = FBO(width, height)    
    fbo.Use()
    glViewport(0, 0, width, height)

    extrinsic_matrices = np.load(extrinsic_matrices_path)
    intrinsics = np.load(intrinsics_path)

    intrinsic_matrix = intrinsics["camMatrix"]
    distortion_coefficients = intrinsics["distCoef"]

    shader = Shader(object_shader["vertex_shader"], object_shader["fragment_shader"])
    object = Object(shader, GetObjProps(obj_file_path))
    camera = Camera(height, width)

    frame_number = 1
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    background_colour = [0.0, 0.0, 0.0]

    for i in tqdm(range(n_frames), desc="Processing frames"):
        # Set viewport and clear
        fbo.Use()
        glClearColor(*background_colour, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # ---rendering---
        camera.Update(shader, intrinsic_matrix, extrinsic_matrices[str(frame_number).zfill(4)])
        object.Draw()

        # Read pixels into NumPy array
        image = fbo.ReadColourBuffer()

        # To use later when adding occlusions
        depth_map = fbo.ReadDepthBuffer()

        # Create mask where rendered image is not the background color
        mask = np.all(image != background_colour, axis=-1)  # shape: (H, W)

        # Create output frame, initially copy original frame
        final_frame = frame.copy()

        # Replace pixels in frame with rendered image where mask is True
        final_frame[mask] = image[mask]

        # final_frame = image.copy() # valo
        cv2.imwrite(f"{output_path}/tmp/output_{frame_number}.png", final_frame)

        ret, frame = cap.read()
        
        frame_number += 1
        
        if not ret:
            break
    
    cap.release()
    images_to_video(f"{output_path}/tmp", f"{output_path}/output.mp4", fps=30)

    shutil.rmtree(f"{output_path}/tmp")

    glfw.destroy_window(window)
    glfw.terminate()