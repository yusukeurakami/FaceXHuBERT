import gc
import os
import pickle as pkl

import cv2
import ffmpeg
import numpy as np
import pyrender
import trimesh


# Transform GT data to match template coordinate system
def transform_gt_to_template_space(gt_data, template_vertices):
    """
    Transform GT data to match the template coordinate system.

    Based on analysis, GT data needs:
    1. Y-axis flip (fixes upside-down issue)
    2. Z-axis flip (fixes depth orientation)
    3. Scale and center matching
    """
    # Apply required axis flips: [1, -1, -1] = [X, -Y, -Z]
    gt_flipped = gt_data.copy()
    gt_flipped[:, 1] *= -1  # Flip Y axis (fixes upside-down)
    gt_flipped[:, 2] *= -1  # Flip Z axis (fixes depth)

    # Center both datasets
    gt_center = np.mean(gt_flipped, axis=0)
    template_center = np.mean(template_vertices, axis=0)

    gt_centered = gt_flipped - gt_center
    template_centered = template_vertices - template_center

    # Scale GT to match template size
    gt_scale = np.max(np.linalg.norm(gt_centered, axis=1))
    template_scale = np.max(np.linalg.norm(template_centered, axis=1))

    if gt_scale > 0:
        scale_factor = template_scale / gt_scale
        gt_scaled = gt_centered * scale_factor
    else:
        gt_scaled = gt_centered

    # Move to template center
    gt_final = gt_scaled + template_center

    return gt_final


def main():
    gt_folder = "./Evaluation/GroundTruth/"
    audio_folder = "./BIWI/wav/"
    gt_seq_name = "F1_e39.npy"
    audio_name = "F1_e39.wav"

    # output folder
    video_folder = "gt_videos/"
    frames_folder = "gt_videos/frames/"
    os.makedirs(video_folder, exist_ok=True)
    os.makedirs(frames_folder, exist_ok=True)

    fps = 25
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
    camera_pose = np.array([[1.0, 0, 0.0, 0.00], [0.0, -1.0, 0.0, 0.00], [0.0, 0.0, 1.0, -1.6], [0.0, 0.0, 0.0, 1.0]])

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)

    r = pyrender.OffscreenRenderer(1920, 1440)

    video_without_audio = video_folder + "gt_without_audio.mp4"
    video_with_audio = video_folder + "gt_with_audio.mp4"

    video = cv2.VideoWriter(video_without_audio, fourcc, fps, (1920, 1440))

    gt_seq_path = gt_folder + gt_seq_name
    audio_path = audio_folder + audio_name
    gt_seq = np.load(gt_seq_path)
    gt_seq = np.reshape(gt_seq, (-1, 70110 // 3, 3))

    # render_mesh = trimesh.load_mesh("./BIWI/templates/BIWI_topology.obj", process=False)
    topology_mesh = trimesh.load_mesh("./BIWI/templates/BIWI_topology.obj", process=False)

    template_data = None
    with open("BIWI/templates_scaled.pkl", "rb") as f:
        template_data = pkl.load(f)

    render_mesh = trimesh.Trimesh(vertices=template_data["F1"], faces=topology_mesh.faces)
    template_vertices = render_mesh.vertices

    # Apply coordinate transformation to GT sequence
    gt_seq_transformed = np.zeros_like(gt_seq)
    for f in range(gt_seq.shape[0]):
        gt_seq_transformed[f] = transform_gt_to_template_space(gt_seq[f], template_vertices)

    render_mesh.vertices = gt_seq_transformed[0, :, :]
    py_mesh = pyrender.Mesh.from_trimesh(render_mesh)

    for f in range(gt_seq.shape[0]):
        render_mesh.vertices = gt_seq_transformed[f, :, :]
        py_mesh = pyrender.Mesh.from_trimesh(render_mesh)
        scene = pyrender.Scene()
        scene.add(py_mesh)

        scene.add(cam, pose=camera_pose)
        scene.add(light, pose=camera_pose)
        color, _ = r.render(scene)
        output_frame = frames_folder + "frame" + str(f) + ".jpg"
        cv2.imwrite(output_frame, color)
        frame = cv2.imread(output_frame)
        video.write(frame)
    video.release()

    input_video = ffmpeg.input(video_without_audio)
    input_audio = ffmpeg.input(audio_path)
    ffmpeg.concat(input_video, input_audio, v=1, a=1).output(video_with_audio).run()
    del video
    gc.collect()


if __name__ == "__main__":
    main()
