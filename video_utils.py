import gc
import os
import pickle as pkl
from typing import Optional, Tuple

import cv2
import ffmpeg
import numpy as np
import pyrender
import trimesh
from tqdm import tqdm

from gt_renderer import transform_gt_to_template_space  # NOQA: F401


class VideoRenderer:
    """
    A utility class for rendering 3D facial animations to video.
    Consolidates rendering functionality used across multiple scripts.
    """

    def __init__(
        self,
        fps: float = 25.0,
        resolution: Tuple[int, int] = (1920, 1440),
        template_path: str = "BIWI/templates_scaled.pkl",
        topology_path: str = "BIWI/templates/BIWI_topology.obj",
        apply_transform: bool = False,
        zoom_factor: float = 1.0,
        camera_distance: float = -1.6,
        dataset_type: str = "BIWI",
    ):
        """
        Initialize the video renderer.

        Args:
            fps: Frame rate for video output
            resolution: Video resolution as (width, height)
            template_path: Path to the template data pickle file
            topology_path: Path to the topology OBJ file
            apply_transform: Whether to apply coordinate transformation
            zoom_factor: Zoom factor for field of view (>1.0 = zoom in, <1.0 = zoom out)
            camera_distance: Distance of camera from object (negative values = closer)
            dataset_type: Dataset type ("BIWI" or "VOCASET")
        """
        self.fps = fps
        self.resolution = resolution
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.apply_transform = apply_transform
        self.dataset_type = dataset_type

        # Auto-configure paths based on dataset type if not explicitly provided
        if template_path == "BIWI/templates_scaled.pkl" and dataset_type == "VOCASET":
            template_path = "VOCASET/templates/templates.pkl"
            topology_path = "VOCASET/templates/FLAME_sample.ply"

        # Load templates and topology based on dataset type
        self.load_templates_and_topology(template_path, topology_path)

        # Set up pyrender components with zoom support
        base_fov = np.pi / 3.0
        adjusted_fov = base_fov / zoom_factor
        self.cam = pyrender.PerspectiveCamera(yfov=adjusted_fov, aspectRatio=1.414)
        self.camera_pose = np.array(
            [[1.0, 0, 0.0, 0.00], [0.0, -1.0, 0.0, 0.00], [0.0, 0.0, 1.0, camera_distance], [0.0, 0.0, 0.0, 1.0]]
        )
        self.light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)
        self.renderer = pyrender.OffscreenRenderer(*resolution)

    def load_templates_and_topology(self, template_path: str, topology_path: str):
        """Load templates and topology based on dataset type."""
        if self.dataset_type == "BIWI":
            # Load pickle templates + OBJ topology
            with open(template_path, 'rb') as f:
                templates = pkl.load(f, encoding='latin1')
                # Ensure templates are flattened
                self.template_data = {}
                for subject, template in templates.items():
                    if len(template.shape) > 1:
                        self.template_data[subject] = template.flatten()
                    else:
                        self.template_data[subject] = template
            self.topology_mesh = trimesh.load_mesh(topology_path, process=False)
        elif self.dataset_type == "VOCASET":
            # Load pickle templates like BIWI
            with open(template_path, 'rb') as f:
                templates = pkl.load(f, encoding='latin1')
                # Ensure templates are flattened
                self.template_data = {}
                for subject, template in templates.items():
                    if len(template.shape) > 1:
                        self.template_data[subject] = template.flatten()
                    else:
                        self.template_data[subject] = template
            # Load topology mesh separately
            self.topology_mesh = trimesh.load_mesh(topology_path, process=False)
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

    def render_sequence_to_video(
        self,
        prediction_path: str,
        subject: str,
        output_video_path: str,
        frames_folder: str,
        audio_path: Optional[str] = None,
        output_with_audio_path: Optional[str] = None,
    ):
        """
        Render a prediction sequence to video.

        Args:
            prediction_path: Path to the .npy prediction file
            subject: Subject identifier for template selection
            output_video_path: Output path for video without audio
            frames_folder: Directory to save individual frames
            audio_path: Optional path to audio file to combine with video
            output_with_audio_path: Optional output path for video with audio
        """
        # Ensure output directories exist
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        os.makedirs(frames_folder, exist_ok=True)
        if output_with_audio_path:
            os.makedirs(os.path.dirname(output_with_audio_path), exist_ok=True)

        # Load prediction data - dataset-aware reshaping
        seq = np.load(prediction_path)
        if self.dataset_type == "BIWI":
            seq = np.reshape(seq, (-1, 70110 // 3, 3))  # 23370 vertices
            # Use subject-specific template
            if subject in self.template_data:
                template_vertices = self.template_data[subject].reshape(-1, 3)
            else:
                raise ValueError(f"Template not found for subject {subject} in BIWI dataset")
        elif self.dataset_type == "VOCASET":
            seq = np.reshape(seq, (-1, 15069 // 3, 3))  # 5023 vertices
            # Use subject-specific template with _TA suffix or default
            subject_with_ta = subject + "_TA"
            if subject_with_ta in self.template_data:
                template_vertices = self.template_data[subject_with_ta].reshape(-1, 3)
            elif subject in self.template_data:
                template_vertices = self.template_data[subject].reshape(-1, 3)
            elif "default" in self.template_data:
                template_vertices = self.template_data["default"].reshape(-1, 3)
            else:
                raise ValueError(f"Template not found for subject {subject} in VOCASET dataset")
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

        # # Create reference mesh using template vertices and topology faces
        # ref_mesh = trimesh.Trimesh(vertices=template_vertices, faces=self.topology_mesh.faces)
        # self.template_vertices = ref_mesh.vertices

        ref_mesh = trimesh.Trimesh(vertices=template_vertices, faces=self.topology_mesh.faces)
        if self.dataset_type == "VOCASET":
            ref_mesh.vertices = transform_gt_to_template_space(ref_mesh.vertices, self.topology_mesh.vertices)
        self.template_vertices = ref_mesh.vertices

        # Transform sequence to template space (only if needed)
        # TODO: Voca needs True, BIWI needs False
        if True:  # self.apply_transform:
            seq_transformed = np.zeros_like(seq)
            for f in range(seq.shape[0]):
                seq_transformed[f] = transform_gt_to_template_space(seq[f], self.template_vertices)
        else:
            seq_transformed = seq

        # Initialize video writer
        print(f"Initializing video writer for {output_video_path}")
        video = cv2.VideoWriter(output_video_path, self.fourcc, self.fps, self.resolution)

        # Render each frame
        for f in tqdm(range(seq.shape[0]), desc="Rendering frames"):
            ref_mesh.vertices = seq_transformed[f, :, :]
            py_mesh = pyrender.Mesh.from_trimesh(ref_mesh)
            scene = pyrender.Scene()
            scene.add(py_mesh)
            scene.add(self.cam, pose=self.camera_pose)
            scene.add(self.light, pose=self.camera_pose)

            color, _ = self.renderer.render(scene)

            # Save frame and add to video
            output_frame = os.path.join(frames_folder, f"frame{f}.jpg")
            cv2.imwrite(output_frame, color)
            frame = cv2.imread(output_frame)
            video.write(frame)

        video.release()
        print(f"Video writer released for {output_video_path}")

        # Verify file was created
        if os.path.exists(output_video_path):
            file_size = os.path.getsize(output_video_path)
            print(f"Video file created successfully: {output_video_path} (Size: {file_size} bytes)")
        else:
            print(f"ERROR: Video file was not created: {output_video_path}")

        # Add audio if provided
        if audio_path and output_with_audio_path:
            print(f"Adding audio to video for {output_with_audio_path}")
            try:
                # Use ffmpeg to combine video and audio
                input_video = ffmpeg.input(output_video_path)
                input_audio = ffmpeg.input(audio_path)
                ffmpeg.concat(input_video, input_audio, v=1, a=1).output(output_with_audio_path).run(
                    overwrite_output=True
                )
                print(f"Audio added successfully to {output_with_audio_path}")
            except Exception as e:
                print(f"Warning: Could not add audio to video. Error: {e}")

        # Clean up
        del video, seq, ref_mesh
        gc.collect()

    def render_prediction_with_naming(
        self,
        prediction_path: str,
        subject: str,
        condition_subject: str,
        base_name: str,
        output_dir: str,
        emotion_label: str = "neutral",
        audio_path: Optional[str] = None,
    ):
        """
        Render a prediction with standardized naming convention.

        Args:
            prediction_path: Path to the .npy prediction file
            subject: Subject identifier for template selection
            condition_subject: Conditioning subject used for prediction
            base_name: Base name for output files (e.g., test file name)
            output_dir: Base output directory
            emotion_label: Emotion label for naming
            audio_path: Optional audio file path
        """
        # Create standardized naming
        output_name = f"{base_name}_{emotion_label}_{subject}_Condition_{condition_subject}"

        # Set up paths
        video_woA_folder = os.path.join(output_dir, "videos_no_audio")
        video_wA_folder = os.path.join(output_dir, "videos_with_audio")
        frames_folder = os.path.join(output_dir, "frames")

        video_woA_path = os.path.join(video_woA_folder, f"{output_name}.mp4")
        video_wA_path = os.path.join(video_wA_folder, f"{output_name}.mp4") if audio_path else None

        print(f"Rendering video for: {output_name}")
        self.render_sequence_to_video(
            prediction_path=prediction_path,
            subject=subject,
            output_video_path=video_woA_path,
            frames_folder=frames_folder,
            audio_path=audio_path,
            output_with_audio_path=video_wA_path,
        )

    def cleanup(self):
        """Clean up renderer resources."""
        if hasattr(self, 'renderer'):
            self.renderer.delete()


def create_video_from_prediction(
    prediction_path: str,
    subject: str,
    condition_subject: str,
    base_name: str,
    output_dir: str,
    emotion_label: str = "neutral",
    audio_path: Optional[str] = None,
    fps: float = 25.0,
    dataset_type: str = "BIWI",
    zoom_factor: float = 1.0,
    camera_distance: float = -1.6,
):
    """
    Convenience function to create a video from a prediction file.

    Args:
        prediction_path: Path to the .npy prediction file
        subject: Subject identifier for template selection
        condition_subject: Conditioning subject used for prediction
        base_name: Base name for output files
        output_dir: Base output directory
        emotion_label: Emotion label for naming
        audio_path: Optional audio file path
        fps: Frame rate for video
        dataset_type: Dataset type ("BIWI" or "VOCASET")
        zoom_factor: Zoom factor for field of view (>1.0 = zoom in, <1.0 = zoom out)
        camera_distance: Distance of camera from object (negative values = closer)
    """
    renderer = VideoRenderer(
        fps=fps,
        dataset_type=dataset_type,
        apply_transform=True,
        zoom_factor=zoom_factor,
        camera_distance=camera_distance,
    )
    try:
        renderer.render_prediction_with_naming(
            prediction_path=prediction_path,
            subject=subject,
            condition_subject=condition_subject,
            base_name=base_name,
            output_dir=output_dir,
            emotion_label=emotion_label,
            audio_path=audio_path,
        )
    finally:
        renderer.cleanup()
