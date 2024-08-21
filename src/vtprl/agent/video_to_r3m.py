import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from load_pretrained_model import r3m_wrapper
import torchvision.transforms as transforms
from PIL import Image


def save_image(image: np.ndarray, current_update: int):
    filename = str(current_update) + '.jpg'
    print(filename)
    cv2.imwrite(filename, image)


def image_to_video(path_to_images, path_to_video, new_filename):
    image_folder = path_to_images  # Path to the folder containing the JPG images
    video_name = path_to_video + new_filename + '.mp4'  # Name and path of the output video file

    # Get the list of JPG files in the specified folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort(key=lambda x: int(x.split('.')[0])) # Sort the images alphabetically if necessary

    # Define the video codec, output video's frame rate, and resolution
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0  # Frames per second
    frame_size = (600, 600)  # Video frame size (width, height)

    # Create a VideoWriter object to write the video
    video = cv2.VideoWriter(video_name, fourcc, fps, frame_size)

    # Read each image, resize if necessary, and write it to the video
    for image in images:
        img_path = os.path.join(image_folder, image)
        img = cv2.imread(img_path)
        img = cv2.resize(img, frame_size)  # Resize image to match frame size if necessary
        video.write(img)

    # Release the VideoWriter and close any open windows
    video.release()
    cv2.destroyAllWindows()

def video_to_r3m_video(path_to_video_and_filename, path_to_r3m_video, output_filename):

    """
    Translates a video that is stored in path_to_video and applies the r3m embedding (according to r3m_model_resolution)
    and stores the embedding in path_to_r3m_video.
    """

    input_video_name = path_to_video_and_filename  # Path to the folder containing the input video
    video_name = path_to_r3m_video + output_filename +'.mp4'  # Name and path of the output video file

    # Open the video file
    cap = cv2.VideoCapture(input_video_name)

    # Check if video opened successfully
    if (cap.isOpened()== False):
      print("Error opening video stream or file")

    # Get the video's frames per second (fps), frame width, and frame height
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = 640
    frame_height = 320
    frame_size = (frame_width, frame_height)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_name, fourcc, fps, frame_size)

    current_process = 0

    while(cap.isOpened()):
        current_process += 1
        ret, frame = cap.read()  # Read the frame
        if ret == True:
            # Apply the r3m model to the current frame
            r3m_frame = r3m_wrapper(frame)
            r3m_frame = r3m_frame.reshape(32, 64)
            r3m_scaled_image = (r3m_frame * 255)

            # Create a figure object and set the size in inches
            fig = plt.figure(figsize=(64, 32))  # Width: 8 inches, Height: 6 inches

            # Adjust the spacing between the plot and the figure edges
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

            plt.imshow(r3m_scaled_image, cmap='gray')
            plt.axis('off')  # Turn off axis
            fig.savefig(path_to_r3m_video + 'temp' + '.jpg', dpi=8)
            plt.close()

            # Read the image file using cv2.imread()
            grayscale_r3m_frame = cv2.imread(path_to_r3m_video + 'temp' + '.jpg')
            grayscale_r3m_frame = cv2.resize(grayscale_r3m_frame, frame_size)  #Resize image to match frame size if necessary (frame_width, frame_height)

            # Write the transformed frame into the file
            out.write(grayscale_r3m_frame)

        # Break the loop if no more frames
        else:
            break

    # Release everything when the job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


input_file_path = '/home/autumn/grasping/src/vtprl/data/input/camera_position_1.mp4'
output_video_path = '/home/autumn/grasping/src/vtprl/data/output/'
filename = 'cam_pos1_r3m'
video_to_r3m_video(input_file_path, output_video_path, filename)
