import cv2
from moviepy.editor import VideoFileClip
import glob


def video_to_images(video_path, images_path):
    clip = VideoFileClip(video_path)
    clip.write_images_sequence(nameformat="{}/img%04d.jpg".format(images_path))


def images_to_video(images_path, video_path):
    images = glob.glob("{}/*.jpg".format(images_path))
    image = cv2.imread(images[0])
    shape = image.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (shape[1], shape[0]))

    for path in images:
        image = cv2.imread(path)
        out.write(image)

    out.release()