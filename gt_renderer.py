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


def main(
    dataset_type="BIWI",
    gt_seq_name=None,
    audio_name=None,
    subject_id="F1",
    gt_folder=None,
    audio_folder=None,
    video_folder=None,
    zoom_factor=1.0,
    camera_distance=-1.6,
):
    """
    Render ground truth sequences for facial animation.

    Args:
        dataset_type (str): Dataset type - "BIWI" or "VOCASET"
        gt_seq_name (str): Name of the ground truth sequence file (without extension for VOCASET)
        audio_name (str): Name of the audio file
        subject_id (str): Subject ID for BIWI dataset (e.g., "F1", "M1")
        gt_folder (str): Path to ground truth folder
        audio_folder (str): Path to audio folder
        video_folder (str): Output video folder
        zoom_factor (float): Zoom factor for field of view (>1.0 = zoom in, <1.0 = zoom out)
        camera_distance (float): Distance of camera from object (negative values = closer)
    """

    # Set default values based on dataset type
    if dataset_type == "BIWI":
        if gt_folder is None:
            gt_folder = "./Evaluation/GroundTruth/"
        if audio_folder is None:
            audio_folder = "./BIWI/wav/"
        if gt_seq_name is None:
            gt_seq_name = "F1_e39.npy"
        if audio_name is None:
            audio_name = "F1_e39.wav"
        vertice_dim = 70110  # 23370 vertices * 3
        template_path = "./BIWI/templates/BIWI_topology.obj"
        template_data_path = "BIWI/templates_scaled.pkl"
        fps = 25
        # Apply zoom factor to field of view (smaller FOV = zoomed in)
        base_fov = np.pi / 3.0
        adjusted_fov = base_fov / zoom_factor
        cam = pyrender.PerspectiveCamera(yfov=adjusted_fov, aspectRatio=1.414)
        camera_pose = np.array(
            [[1.0, 0, 0.0, 0.00], [0.0, -1.0, 0.0, 0.00], [0.0, 0.0, 1.0, camera_distance], [0.0, 0.0, 0.0, 1.0]]
        )

    elif dataset_type == "VOCASET":
        if gt_folder is None:
            gt_folder = "./VOCASET/vertices_npy/"
        if audio_folder is None:
            audio_folder = "./VOCASET/wav/"
        if gt_seq_name is None:
            gt_seq_name = "FaceTalk_170725_00137_TA_sentence01.npy"
        if audio_name is None:
            audio_name = "FaceTalk_170725_00137_TA_sentence01.wav"
        vertice_dim = 15069  # 5023 vertices * 3
        template_path = "./VOCASET/templates/FLAME_sample.ply"
        template_data_path = None  # VOCASET uses direct PLY template
        fps = 60
        # Apply zoom factor to field of view (smaller FOV = zoomed in)
        base_fov = np.pi / 3.0
        adjusted_fov = base_fov / zoom_factor
        cam = pyrender.PerspectiveCamera(yfov=adjusted_fov, aspectRatio=1.414)
        camera_pose = np.array(
            [[1.0, 0, 0.0, 0.00], [0.0, -1.0, 0.0, 0.00], [0.0, 0.0, 1.0, camera_distance], [0.0, 0.0, 0.0, 1.0]]
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    # Set output folder
    if video_folder is None:
        video_folder = f"gt_videos_{dataset_type.lower()}/"
    frames_folder = video_folder + "frames/"
    os.makedirs(video_folder, exist_ok=True)
    os.makedirs(frames_folder, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)

    r = pyrender.OffscreenRenderer(1920, 1440)

    video_without_audio = video_folder + "gt_without_audio.mp4"
    video_with_audio = video_folder + "gt_with_audio.mp4"

    video = cv2.VideoWriter(video_without_audio, fourcc, fps, (1920, 1440))

    gt_seq_path = gt_folder + gt_seq_name
    audio_path = audio_folder + audio_name
    gt_seq = np.load(gt_seq_path)
    gt_seq = np.reshape(gt_seq, (-1, vertice_dim // 3, 3))

    # Load template and create render mesh based on dataset type
    if dataset_type == "BIWI":
        topology_mesh = trimesh.load_mesh(template_path, process=False)

        template_data = None
        with open(template_data_path, "rb") as f:
            template_data = pkl.load(f)

        render_mesh = trimesh.Trimesh(vertices=template_data[subject_id], faces=topology_mesh.faces)

    elif dataset_type == "VOCASET":
        # Load FLAME template directly
        render_mesh = trimesh.load_mesh(template_path, process=False)

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
    import argparse

    parser = argparse.ArgumentParser(description="Render ground truth sequences for facial animation")
    parser.add_argument(
        "--dataset_type", type=str, default="BIWI", choices=["BIWI", "VOCASET"], help="Dataset type: BIWI or VOCASET"
    )
    parser.add_argument("--gt_seq_name", type=str, default=None, help="Ground truth sequence file name")
    parser.add_argument("--audio_name", type=str, default=None, help="Audio file name")
    parser.add_argument("--subject_id", type=str, default="F1", help="Subject ID for BIWI dataset (e.g., F1, M1)")
    parser.add_argument("--gt_folder", type=str, default=None, help="Path to ground truth folder")
    parser.add_argument("--audio_folder", type=str, default=None, help="Path to audio folder")
    parser.add_argument("--video_folder", type=str, default=None, help="Output video folder")
    parser.add_argument(
        "--zoom_factor", type=float, default=1.0, help="Zoom factor for field of view (>1.0 = zoom in, <1.0 = zoom out)"
    )
    parser.add_argument(
        "--camera_distance", type=float, default=-1.6, help="Distance of camera from object (negative values = closer)"
    )

    args = parser.parse_args()

    # Call main with parsed arguments
    main(
        dataset_type=args.dataset_type,
        gt_seq_name=args.gt_seq_name,
        audio_name=args.audio_name,
        subject_id=args.subject_id,
        gt_folder=args.gt_folder,
        audio_folder=args.audio_folder,
        video_folder=args.video_folder,
        zoom_factor=args.zoom_factor,
        camera_distance=args.camera_distance,
    )

    print("\n=== Usage Examples ===")
    print("# For BIWI dataset (default):")
    print("python gt_renderer.py --dataset_type BIWI --subject_id F1 --gt_seq_name F1_e39.npy --audio_name F1_e39.wav")
    print("\n# For VOCASET dataset:")
    print(
        "python gt_renderer.py --dataset_type VOCASET --gt_seq_name FaceTalk_170904_03276_TA_sentence01.npy --audio_name FaceTalk_170904_03276_TA_sentence01.wav"
    )
    print("\n# With zoom functionality:")
    print("python gt_renderer.py --dataset_type BIWI --subject_id F1 --zoom_factor 2.0 --camera_distance -1.2")
    print("\n# With custom paths:")
    print(
        "python gt_renderer.py --dataset_type VOCASET --gt_folder ./custom_gt/ --audio_folder ./custom_audio/ --video_folder ./custom_output/"
    )
    print("\n=== Zoom Parameters ===")
    print("--zoom_factor: Controls field of view (>1.0 = zoom in, <1.0 = zoom out)")
    print("  • 1.0 = normal view")
    print("  • 2.0 = 2x zoom in (recommended for close-up)")
    print("  • 0.5 = 2x zoom out (wider view)")
    print("--camera_distance: Controls camera position (-1.6 = default)")
    print("  • -1.2 = closer to object")
    print("  • -2.0 = further from object")
