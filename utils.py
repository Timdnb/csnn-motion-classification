import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import Image

def create_sample(shape, motion, frame_size, n_frames):
    """ Creates a sample, which is a set of frames of a shape with a certain motion.

    Args:
        shape (str): The shape of the object to create. Either "circle" or "square".
        motion (str): The motion of the object. Either "up", "down", "left", "right" or "rotation".
        frame_size (int): The size of the frames.
        n_frames (int): The number of frames to create.

    Returns:
        np.array: The frames of the sample.
    """
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
    """ Convert frames to an event-based dataset.

    Args:
        frames (np.array): The frames to convert.

    Returns:
        np.array: The event-based dataset.
    """
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

def spiking_overview(spks, events, input_dim, filename, h_dist_between_graphs=32, v_dist_between_graphs=4, output_dim=5):
    """ Takes the spiking activity of the network and the event-based dataset and creates an animation. 

    Args:
        spks (list): list of outputs for every time step for every spiking layer
        events (np.array): the event-based dataset
        input_dim (int): the dimension of the input events
        filename (str): the filename to save the animation as
        h_dist_between_graphs (int, optional): horizontal distance between the graphs. Defaults to 32.
        v_dist_between_graphs (int, optional): vertical distance between the graphs. Defaults to 4.
        output_dim (int, optional): dimension of the output spikes. Defaults to 5.

    Returns:
        Image: the animation
    """
    # create lists
    widths = []
    heigths = []

    # loop through all spiking layers except the last one
    for spk in spks[:-1]:
        widths.append(spk.shape[3])
        heigths.append(spk.shape[1]*spk.shape[2] + v_dist_between_graphs*(spk.shape[1]-1))

    # calculate canvas size based on the spiking layers
    canvas_width = sum(widths) + h_dist_between_graphs*(len(widths)+1) + input_dim + output_dim
    canvas_height = max(heigths)

    # create figure and ims list
    fig = plt.figure(figsize=(15*(canvas_width/canvas_height), 15))
    ims = []

    # iterate through time
    for step in range(spks[0].shape[0]):
        # create canvas
        canvas = np.ones((canvas_height, canvas_width))*2

        # first add the inputs (e.g the events)
        y_top_left = (canvas_height-input_dim)//2
        canvas[y_top_left:y_top_left+input_dim, :input_dim] = events[step].squeeze()

        # then loop through spiking layers except the last one
        for i, spk in enumerate(spks[:-1]):
            y_start = (canvas_height-heigths[i])//2
            x_left = input_dim + (i+1)*h_dist_between_graphs + sum(widths[:i])
            for j in range(spk.shape[1]):
                y_top_left = j*(spk.shape[2]+v_dist_between_graphs) + y_start
                canvas[y_top_left:y_top_left+spk.shape[2], x_left:x_left+spk.shape[3]] = spk[step, j]
        
        # finally add the output spikes
        x_left = input_dim + (len(spks))*h_dist_between_graphs + sum(widths)
        n_outputs = spks[-1].shape[1]
        y_start = (canvas_height-n_outputs*output_dim-(n_outputs-1)*v_dist_between_graphs)//2
        for j in range(n_outputs):
            y_top_left = j*(output_dim+v_dist_between_graphs) + y_start
            mini_canvas = np.ones((output_dim, output_dim))*spks[-1][step, j]
            canvas[y_top_left:y_top_left+output_dim, x_left:x_left+output_dim] = mini_canvas

        # define a custom colormap
        colors = [(0, 0, 0), (0.5, 0.5, 0.5), (1, 1, 1), (0.9, 0.9, 0.9)]
        n_bins = 100
        cmap_name = 'custom_cmap'
        cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

        # create frame and add frame number
        im = plt.imshow(canvas, animated=True, vmin=-1, vmax=2, cmap=cmap, interpolation='none')
        text = plt.text(10, -10, f'Frame: {step}', color='white', bbox=dict(facecolor='black', alpha=0.5))
        ims.append([im, text])

    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=1000)

    # save the animation
    ani.save(f"animations/{filename}.gif", writer="pillow")
    plt.close()
    return Image(filename=f"animations/{filename}.gif")

if __name__ == "__main__":
    pass