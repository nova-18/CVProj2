import streamlit as st
import os
from pathlib import Path
import shutil
import base64
from calibrate import CalibrateCamera
from marker import EstimateExtrinsicUsingMarker
from render import Render
import subprocess
import time
# from moviepy.editor import VideoFileClip


def start_xvfb():
    """Starts Xvfb and sets the DISPLAY environment variable."""
    try:
        # Start Xvfb in the background
        subprocess.Popen(["Xvfb", ":99", "-screen", "0", "1024x768x24"])
        time.sleep(2) # Give Xvfb a moment to start.
        os.environ["DISPLAY"] = ":99" #This affects the streamlit process.
        st.success("Xvfb started and DISPLAY set (Streamlit only).")

    except Exception as e:
        st.error(f"Error starting Xvfb: {e}")

def kill_xvfb():
    """Kills the Xvfb process."""
    try:
        subprocess.run(["pkill", "Xvfb"], shell=True, check=True)
        st.success("Xvfb process killed.")
    except subprocess.CalledProcessError as e:
        st.error(f"Error killing Xvfb: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

def set_libgl_software():
    """Sets the LIBGL_ALWAYS_SOFTWARE environment variable."""
    try:
        # Note: Directly setting environment variables within a subprocess this way
        # typically will *not* affect the parent process (streamlit).
        # This only affects the environment of the subprocess itself.
        # If your goal is to affect the Streamlit process, this will not work.
        subprocess.run(["export", "LIBGL_ALWAYS_SOFTWARE=1"], shell=True, check=True)
        st.success("LIBGL_ALWAYS_SOFTWARE set (in subprocess).")

    except subprocess.CalledProcessError as e:
        st.error(f"Error setting LIBGL_ALWAYS_SOFTWARE: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

def re_encode_video(input_file, output_file):
    """Re-encodes a video using FFmpeg."""
    try:
        command = [
            "ffmpeg",
            "-i", input_file,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-movflags", "+faststart",
            output_file,
        ]
        subprocess.run(command, check=True, capture_output=True)
        return True, None  # Success, no error
    except subprocess.CalledProcessError as e:
        return False, f"FFmpeg error: {e.stderr.decode()}"
    except FileNotFoundError:
        return False, "FFmpeg not found. Please install FFmpeg."
    except Exception as e:
        return False, f"An unexpected error occurred: {e}"

def wait(seconds):
    """Wait for a specified number of seconds."""
    import time
    time.sleep(seconds)

# Set up directories
set_libgl_software()
start_xvfb()
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("data/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

scroll_script = """
<script>
window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});
</script>
"""

# Sidebar options
st.sidebar.header("Configuration")
position_method = st.sidebar.selectbox("Positioning Method", ["Marker", "Markerless"])
depth_method = st.sidebar.selectbox("Depth Estimation", ["Stereo", "SfM"])

# Main UI
st.title("Computer Vision Project UI")

# File uploaders
st.subheader("Upload Files")
CALIB_VIDEO_PATH = UPLOAD_DIR / "calibration_video.mp4"
MARKER_VIDEO_PATH = UPLOAD_DIR / "marker_video.mp4"
OBJ_FILE_PATH = UPLOAD_DIR / "model.obj"
calib_video = st.file_uploader("Upload Calibration Video", type=["mp4"], key="calib", label_visibility='collapsed')
if calib_video:
    with open(CALIB_VIDEO_PATH, "wb") as f:
        f.write(calib_video.read())
    st.video(str(CALIB_VIDEO_PATH))


marker_video = st.file_uploader("Upload Marker Video", type=["mp4"], key="marker", label_visibility='collapsed')
if marker_video:
    with open(MARKER_VIDEO_PATH, "wb") as f:
        f.write(marker_video.read())
    st.video(str(MARKER_VIDEO_PATH))


obj_file = st.file_uploader("Upload OBJ File", type=["obj"], key="obj", label_visibility='collapsed')
if obj_file:
    with open(OBJ_FILE_PATH, "wb") as f:
        f.write(obj_file.read())
    st.success("OBJ file uploaded.")


# Save uploaded files



submit = st.button("Submit")

if submit:
    st.subheader("Pipeline Progress")

    square_size = 19.00
    n_images = 40

    # Step 1: Calibration
    with st.spinner("Running Calibration..."):
        try:
            CalibrateCamera(str(CALIB_VIDEO_PATH), str(OUTPUT_DIR), square_size, n_images)
            # wait(2)
            st.success("Calibration Complete ✅")
        except Exception as e:
            st.error(f"Calibration Failed ❌: {e}")

    # Step 2: Estimating Extrinsics
    with st.spinner("Estimating Camera Extrinsics..."):
        try:
            EstimateExtrinsicUsingMarker(
                str(MARKER_VIDEO_PATH),
                str(OUTPUT_DIR / "calib_data.npz"),
                str(OUTPUT_DIR / "extrinsics.npz"),
                str(OUTPUT_DIR / "intermidiate")
            )
            # wait(2)
            st.success("Extrinsics Estimation Complete ✅")
        except Exception as e:
            st.error(f"Extrinsics Estimation Failed ❌: {e}")

    # Step 3: Rendering
    with st.spinner("Rendering Output Video..."):
        try:
            Render(
                str(MARKER_VIDEO_PATH),
                None,
                str(OUTPUT_DIR),
                str(OUTPUT_DIR / "extrinsics.npz"),
                str(OUTPUT_DIR / "calib_data.npz"),
                str(OBJ_FILE_PATH)
            )
            # wait(2)
            output_video_path = "data/output/output.mp4"
            # Load the input video
            # clip = VideoFileClip(output_video_path)

# Write it to a new file with H.264 video and AAC audio codecs
            

            final_out_path = "data/output/output_fin.mp4"
            # clip.write_videofile(final_out_path, codec="libx264", audio_codec="aac")
            # final_out_path = output_video_path
            kill_xvfb()
            success, error_message = re_encode_video(output_video_path,final_out_path)
            if os.path.exists(final_out_path):
                st.success("Rendering Complete ✅")
                st.video(str(final_out_path))
                with open(final_out_path, "rb") as f:
                    video_bytes = f.read()
                    b64 = base64.b64encode(video_bytes).decode()
                    href = f'<a href="data:video/mp4;base64,{b64}" download="output_video.mp4">Download Output Video</a>'
                    st.markdown(href, unsafe_allow_html=True)
            else:
                st.error("Output video not found ❌")
        except Exception as e:
            st.error(f"Rendering Failed ❌: {e}")

st.markdown(scroll_script, unsafe_allow_html=True)
