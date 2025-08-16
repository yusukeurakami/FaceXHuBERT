import gc
import os
import pickle as pkl
from typing import Optional, Tuple

import cv2
import ffmpeg
import numpy as np
import pyrender
import trimesh

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
    ):
        """
        Initialize the video renderer.

        Args:
            fps: Frame rate for video output
            resolution: Video resolution as (width, height)
            template_path: Path to the template data pickle file
            topology_path: Path to the topology OBJ file
        """
        self.fps = fps
        self.resolution = resolution
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.apply_transform = apply_transform

        # Load template data
        with open(template_path, 'rb') as f:
            self.template_data = pkl.load(f, encoding='latin1')

        # Load topology mesh
        self.topology_mesh = trimesh.load_mesh(topology_path, process=False)

        # Set up pyrender components
        self.cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
        self.camera_pose = np.array(
            [[1.0, 0, 0.0, 0.00], [0.0, -1.0, 0.0, 0.00], [0.0, 0.0, 1.0, -1.6], [0.0, 0.0, 0.0, 1.0]]
        )
        self.light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)
        self.renderer = pyrender.OffscreenRenderer(*resolution)

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

        # Load prediction data
        seq = np.load(prediction_path)
        seq = np.reshape(seq, (-1, 70110 // 3, 3))

        # Create reference mesh using template vertices and topology faces
        ref_mesh = trimesh.Trimesh(vertices=self.template_data[subject], faces=self.topology_mesh.faces)
        self.template_vertices = ref_mesh.vertices

        # Transform sequence to template space
        if self.apply_transform:
            seq_transformed = np.zeros_like(seq)
            for f in range(seq.shape[0]):
                seq_transformed[f] = transform_gt_to_template_space(seq[f], self.template_vertices)
        else:
            seq_transformed = seq

        # Initialize video writer
        video = cv2.VideoWriter(output_video_path, self.fourcc, self.fps, self.resolution)

        # Render each frame
        for f in range(seq.shape[0]):
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

        # Add audio if provided
        if audio_path and output_with_audio_path:
            try:
                input_video = ffmpeg.input(output_video_path)
                input_audio = ffmpeg.input(audio_path)
                ffmpeg.concat(input_video, input_audio, v=1, a=1).output(output_with_audio_path).run(
                    overwrite_output=True
                )
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
    """
    renderer = VideoRenderer(fps=fps)
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
