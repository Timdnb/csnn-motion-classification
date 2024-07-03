import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import Image

def create_sample(shape, motion, frame_size, n_frames):
    # Generate random shape size and position
    shape_size = np.random.randint(int(0.1*frame_size), int(0.2*frame_size))
    all_labels = ["up", "down", "left", "right", "rotation"]
    label = all_labels.index(motion)
    
    # Get start and end position for the motion
    if motion == "up":
        start_position = (np.random.randint(-shape_size, frame_size+shape_size), np.random.randint(int(frame_size*0.3), frame_size+shape_size))
        end_position = (start_position[0], -shape_size)
    elif motion == "down":
        start_position = (np.random.randint(-shape_size, frame_size+shape_size), np.random.randint(-shape_size, int(frame_size*0.7)))
        end_position = (start_position[0], frame_size+shape_size)
    elif motion == "left":
        start_position = (np.random.randint(int(frame_size*0.3), frame_size+shape_size), np.random.randint(-shape_size, frame_size+shape_size))
        end_position = (-shape_size, start_position[1])
    elif motion == "right":
        start_position = (np.random.randint(-shape_size, int(frame_size*0.7)), np.random.randint(-shape_size, frame_size+shape_size))
        end_position = (frame_size+shape_size, start_position[1])
    elif motion == "rotation":
        start_position = (np.random.randint(-shape_size, frame_size+shape_size), np.random.randint(-shape_size, frame_size+shape_size))
        end_position = start_position
        max_rotation = np.random.uniform(1*np.pi, 4*np.pi)

    # print(f"Shape: {shape}, Motion: {motion}, Start: {start_position}, End: {end_position}, Size: {shape_size}")

    # Generate frames
    frames = np.empty((n_frames, frame_size, frame_size), dtype=np.uint8)

    for step in range(n_frames):
        # Create blank canvas
        canvas = np.zeros((frame_size, frame_size), dtype=np.uint8)

        # Calculate the position of the shape for the current frame
        t = step / n_frames
        position = (int(start_position[0] + (end_position[0] - start_position[0]) * t),
                    int(start_position[1] + (end_position[1] - start_position[1]) * t))
        
        # Draw the shape
        if shape == "circle":
            if motion == "rotation":
                angle = max_rotation * t
                position = (position[0] + 2*shape_size, position[1] + 2*shape_size)
                rot_mat = cv2.getRotationMatrix2D(position, np.degrees(angle), 1.0)

                # Create a larger canvas to avoid cropping during rotation
                canvas = np.zeros((frame_size+4*shape_size, frame_size+4*shape_size), dtype=np.uint8)
                frame = cv2.circle(canvas, position, shape_size, 255, -1)
                frame = cv2.warpAffine(frame, rot_mat, canvas.shape, flags=cv2.INTER_LINEAR)
                frame = frame[2*shape_size:2*shape_size+frame_size, 2*shape_size:2*shape_size+frame_size]
            else:
                frame = cv2.circle(canvas, position, shape_size, 255, -1)
        elif shape == "square":
            if motion == "rotation":
                angle = max_rotation * t
                position = (position[0] + 2*shape_size, position[1] + 2*shape_size)
                rot_mat = cv2.getRotationMatrix2D(position, np.degrees(angle), 1.0)
                canvas = np.zeros((frame_size+4*shape_size, frame_size+4*shape_size), dtype=np.uint8)
                rect = (position[0] - shape_size, position[1] - shape_size, shape_size*2, shape_size*2)
                frame = cv2.rectangle(canvas, rect, 255, -1)
                frame = cv2.warpAffine(frame, rot_mat, canvas.shape, flags=cv2.INTER_LINEAR)
                frame = frame[2*shape_size:2*shape_size+frame_size, 2*shape_size:2*shape_size+frame_size]
            else:
                rect = (position[0] - shape_size, position[1] - shape_size, shape_size*2, shape_size*2)
                frame = cv2.rectangle(canvas, rect, 255, -1)

        # frame = np.clip(frame, 0, 1)
        frames[step] = frame

    return frames, label

def make_event_based(frames):
    frame_size = frames.shape[1]
    n_frames = frames.shape[0]

    events = np.empty((n_frames-1, frame_size, frame_size), dtype=np.int8)
    for step in range(1, n_frames):
        frame = frames[step]
        prev_frame = frames[step-1]

        # Subtract the two frames and create event-based dataset
        event = np.zeros((frame_size, frame_size), dtype=np.int8)
        event[frame > prev_frame] = 1
        event[frame < prev_frame] = -1

        events[step-1] = event

    return events

def animate(frames, filename):
    """ Animates the frames and saves the animation as a GIF.

    Args:
        frames (np.array): The frames to animate.
        filename (str): The filename to save the animation as.

    Returns:
        Image: The animation.
    """
    # Get the number of frames
    n_frames = frames.shape[0]

    # Define a custom colormap
    colors = [(1, 0, 0), (0.5, 0.5, 0.5), (0, 1, 0)]  # Red -> Gray -> Green
    n_bins = 100  # Discretize the colormap into 100 bins
    cmap_name = 'custom_cmap'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # Create the animation
    fig = plt.figure()
    ims = []
    for i in range(n_frames):
        im = plt.imshow(frames[i], animated=True, vmin=-1, vmax=1, cmap=cmap)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)

    # Save the animation
    ani.save(f"animations/{filename}", writer="pillow")
    plt.close()
    return Image(filename=f"animations/{filename}")

if __name__ == "__main__":
    pass