#!/usr/bin/env python3
"""
Real-Time Face Animation Application
Captures microphone audio and renders synchronized 3D facial animations
Optimized for RTX4060Ti with high frame rate priority
"""

import argparse
import logging
import os
import queue
import threading
import time
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np
import pyaudio
import torch
import trimesh
from transformers import Wav2Vec2Processor

from dataset_config import auto_configure_args, get_dataset_config
from faceXhubert import FaceXHuBERT
from gt_renderer import transform_gt_to_template_space
from video_utils import VideoRenderer


class RealTimeAudioProcessor:
    """Handles real-time microphone input and audio feature extraction."""

    def __init__(self, sample_rate: int = 16000, chunk_size: int = 800, overlap: int = 400):
        """
        Initialize audio processor.

        Args:
            sample_rate: Audio sampling rate (16kHz for HuBERT)
            chunk_size: Audio chunk size in samples (~50ms at 16kHz)
            overlap: Overlap between chunks for continuity
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.audio_buffer = deque(maxlen=sample_rate * 2)  # 2 second buffer
        self.feature_queue = queue.Queue(maxsize=10)

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.recording = False

        # Initialize HuBERT processor
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-xlarge-ls960-ft")

        # Threading
        self.audio_thread = None
        self.processing_thread = None

        logging.info(f"Audio processor initialized: {sample_rate}Hz, {chunk_size} samples/chunk")

    def start_capture(self):
        """Start audio capture and processing."""
        self.recording = True

        # Open audio stream
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback,
        )

        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_audio_features, daemon=True)
        self.processing_thread.start()

        self.stream.start_stream()
        logging.info("Audio capture started")

    def stop_capture(self):
        """Stop audio capture and processing."""
        self.recording = False

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for incoming audio data."""
        if status:
            logging.warning(f"Audio callback status: {status}")

        # Convert to numpy array and add to buffer
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_buffer.extend(audio_data)

        return (None, pyaudio.paContinue)

    def _process_audio_features(self):
        """Process audio buffer into HuBERT features."""
        last_processed = 0

        while self.recording:
            current_length = len(self.audio_buffer)

            # Check if we have enough new data to process
            if current_length - last_processed >= self.chunk_size - self.overlap:
                # Extract audio chunk with overlap
                start_idx = max(0, last_processed - self.overlap)
                end_idx = min(current_length, start_idx + self.chunk_size)

                if end_idx - start_idx >= self.chunk_size // 2:  # Minimum chunk size
                    # Convert to numpy array
                    audio_chunk = np.array(list(self.audio_buffer))[start_idx:end_idx]

                    # Process with HuBERT
                    try:
                        audio_features = self.processor(
                            audio_chunk, return_tensors="pt", padding="longest", sampling_rate=self.sample_rate
                        ).input_values

                        # Add to feature queue (non-blocking)
                        try:
                            self.feature_queue.put_nowait(
                                {'features': audio_features, 'timestamp': time.time(), 'chunk_length': len(audio_chunk)}
                            )
                        except queue.Full:
                            # Remove oldest if queue is full
                            try:
                                self.feature_queue.get_nowait()
                                self.feature_queue.put_nowait(
                                    {
                                        'features': audio_features,
                                        'timestamp': time.time(),
                                        'chunk_length': len(audio_chunk),
                                    }
                                )
                            except queue.Empty:
                                pass

                        last_processed = end_idx - self.overlap

                    except Exception as e:
                        logging.error(f"Error processing audio features: {e}")

            time.sleep(0.01)  # Small sleep to prevent busy waiting

    def get_latest_features(self) -> Optional[dict]:
        """Get latest audio features if available."""
        try:
            return self.feature_queue.get_nowait()
        except queue.Empty:
            return None

    def cleanup(self):
        """Cleanup audio resources."""
        self.stop_capture()
        if hasattr(self, 'audio'):
            self.audio.terminate()


class FastRenderer:
    """Optimized renderer for high frame rate 3D facial animation."""

    def __init__(self, resolution: Tuple[int, int] = (640, 480), dataset_type: str = "BIWI"):
        """
        Initialize fast renderer.

        Args:
            resolution: Rendering resolution (width, height)
            dataset_type: Dataset type for template loading
        """
        self.resolution = resolution
        self.dataset_type = dataset_type

        # Set zoom factor based on dataset type
        zoom_factor = 1.0 if dataset_type == "BIWI" else 4.0

        # Use VideoRenderer but with optimized settings
        self.renderer = VideoRenderer(
            fps=60,  # High fps target
            resolution=resolution,
            dataset_type=dataset_type,
            zoom_factor=zoom_factor,
            camera_distance=-1.6,
            apply_transform=False,
        )

        # Pre-allocate frame buffer
        self.frame_buffer = np.zeros((*resolution[::-1], 3), dtype=np.uint8)

        logging.info(f"Fast renderer initialized: {resolution[0]}x{resolution[1]}")

    def render_vertices(self, vertices: np.ndarray, subject: str) -> np.ndarray:
        """
        Render 3D vertices to 2D image.

        Args:
            vertices: 3D vertex array (N, 3)
            subject: Subject identifier for template

        Returns:
            Rendered image as numpy array
        """
        try:
            # Use the existing VideoRenderer mesh creation logic
            if self.dataset_type == "BIWI":
                vertices_reshaped = vertices.reshape(-1, 3)
                template_vertices = self.renderer.template_data[subject].reshape(-1, 3)  # NOQA F841
            elif self.dataset_type == "VOCASET":
                vertices_reshaped = vertices.reshape(-1, 3)
                subject_with_ta = subject + "_TA"
                if subject_with_ta in self.renderer.template_data:
                    template_vertices = self.renderer.template_data[subject_with_ta].reshape(-1, 3)  # NOQA F841
                else:
                    template_vertices = self.renderer.template_data[subject].reshape(-1, 3)  # NOQA F841

            # Create mesh
            ref_mesh = trimesh.Trimesh(vertices=vertices_reshaped, faces=self.renderer.topology_mesh.faces)

            # Apply coordinate transformation for VOCASET (similar to video_utils.py)
            if self.dataset_type == "VOCASET":
                # Transform vertices to template space to fix orientation
                ref_mesh.vertices = transform_gt_to_template_space(
                    ref_mesh.vertices, self.renderer.topology_mesh.vertices
                )

            # Render using pyrender (optimized for single frame)
            import pyrender

            py_mesh = pyrender.Mesh.from_trimesh(ref_mesh)
            scene = pyrender.Scene()
            scene.add(py_mesh)
            scene.add(self.renderer.cam, pose=self.renderer.camera_pose)
            scene.add(self.renderer.light, pose=self.renderer.camera_pose)

            color, _ = self.renderer.renderer.render(scene)

            # Convert to BGR for OpenCV
            return cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

        except Exception as e:
            logging.error(f"Rendering error: {e}")
            # Return black frame on error
            return np.zeros((*self.resolution[::-1], 3), dtype=np.uint8)

    def cleanup(self):
        """Cleanup renderer resources."""
        if hasattr(self.renderer, 'renderer'):
            self.renderer.cleanup()


class RealTimeFaceApp:
    """Main application class for real-time face animation."""

    def __init__(self, args):
        """Initialize the real-time face animation application."""
        self.args = args
        self.running = False

        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Initialize components
        self.audio_processor = None
        self.renderer = None
        self.model = None
        self.webcam = None

        # Performance monitoring
        self.fps_counter = deque(maxlen=30)
        self.latency_buffer = deque(maxlen=30)

        # Model parameters
        self.device = torch.device(args.device)
        self.setup_model()

        logging.info("RealTimeFaceApp initialized")

    def setup_model(self):
        """Load and setup the FaceXHuBERT model."""
        logging.info("Loading FaceXHuBERT model...")

        # Load model
        self.model = FaceXHuBERT(self.args)

        # Use dataset-specific model file
        if self.args.dataset_type == "VOCASET":
            model_path = f'pretrained_model/{self.args.model_name}_VOCASET.pth'
        else:
            model_path = f'pretrained_model/{self.args.model_name}_BIWI.pth'

        # Fallback to generic name if specific file doesn't exist
        if not os.path.exists(model_path):
            model_path = f'pretrained_model/{self.args.model_name}.pth'

        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(self.device)
        self.model.eval()

        # Setup templates and conditioning
        config = get_dataset_config(self.args.dataset_type)

        # Use first training subject as default
        train_subjects_list = self.args.train_subjects.split(" ")
        self.subject = train_subjects_list[0]  # Basic/first subject
        self.condition = train_subjects_list[0]  # Same as subject for simplicity

        # Create one-hot encodings
        one_hot_labels = np.eye(len(train_subjects_list))
        self.one_hot = torch.FloatTensor(one_hot_labels[0].reshape(1, -1)).to(self.device)

        # Emotion setting (neutral = 0, emotional = 1)
        emo_one_hot_labels = np.eye(2)
        self.emo_one_hot = torch.FloatTensor(emo_one_hot_labels[0].reshape(1, -1)).to(self.device)  # Neutral

        # Template setup
        import pickle as pkl

        template_file = f"{self.args.dataset_type}/{config['template_file']}"
        with open(template_file, 'rb') as f:
            templates = pkl.load(f, encoding='latin1')

        # Handle VOCASET subject naming (with _TA suffix)
        if self.args.dataset_type == "VOCASET":
            subject_key = self.subject + "_TA" if not self.subject.endswith("_TA") else self.subject
            if subject_key not in templates:
                # Use first available subject if specified one not found
                subject_key = list(templates.keys())[0]
                self.subject = subject_key.replace("_TA", "")
                logging.info(f"Subject not found, using: {self.subject}")
        else:
            subject_key = self.subject

        template = templates[subject_key]
        if len(template.shape) > 1:
            template = template.flatten()
        self.template = torch.FloatTensor(template.reshape(1, -1)).to(self.device)

        logging.info(f"Model loaded successfully. Subject: {self.subject}, Device: {self.device}")

    def setup_components(self):
        """Setup audio processor, renderer, and webcam."""
        # Audio processor
        self.audio_processor = RealTimeAudioProcessor(sample_rate=16000, chunk_size=800, overlap=400)  # ~50ms chunks

        # Renderer (reduced resolution for high fps)
        self.renderer = FastRenderer(resolution=(640, 480), dataset_type=self.args.dataset_type)

        # Webcam setup
        self.webcam = cv2.VideoCapture(0)
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.webcam.set(cv2.CAP_PROP_FPS, 30)

        if not self.webcam.isOpened():
            logging.warning("Webcam not available, using placeholder")
            self.webcam = None

        logging.info("All components setup successfully")

    def predict_vertices(self, audio_features: torch.Tensor) -> Optional[np.ndarray]:
        """
        Predict 3D vertices from audio features.

        Args:
            audio_features: Audio feature tensor from HuBERT

        Returns:
            Predicted vertices as numpy array
        """
        try:
            with torch.no_grad():
                # Ensure proper tensor shapes
                audio_features = audio_features.to(self.device)
                if len(audio_features.shape) == 2:
                    audio_features = audio_features.unsqueeze(0)

                # Model prediction
                prediction = self.model.predict(audio_features, self.template, self.one_hot, self.emo_one_hot)

                # Convert to numpy
                prediction = prediction.squeeze().detach().cpu().numpy()

                # Reshape to vertices
                if self.args.dataset_type == "BIWI":
                    vertices = prediction.reshape(-1, 70110 // 3, 3)
                elif self.args.dataset_type == "VOCASET":
                    vertices = prediction.reshape(-1, 15069 // 3, 3)

                # Return latest frame if sequence
                if len(vertices.shape) == 3:
                    return vertices[-1]  # Last frame
                else:
                    return vertices

        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return None

    def run(self):
        """Main application loop."""
        self.running = True
        self.setup_components()

        # Start audio capture
        self.audio_processor.start_capture()

        # Create display window
        window_name = "Real-Time Face Animation"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 480)  # Side-by-side display

        # Initialize recording if needed
        recording = False
        video_writer = None

        logging.info("Starting real-time processing...")

        try:
            while self.running:
                frame_start = time.time()

                # Get webcam frame
                webcam_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                if self.webcam:
                    ret, frame = self.webcam.read()
                    if ret:
                        webcam_frame = cv2.resize(frame, (640, 480))
                    else:
                        # Webcam placeholder
                        cv2.putText(
                            webcam_frame,
                            "Webcam not available",
                            (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2,
                        )

                # Get latest audio features
                audio_data = self.audio_processor.get_latest_features()
                rendered_frame = np.zeros((480, 640, 3), dtype=np.uint8)

                if audio_data:
                    # Process audio features
                    audio_features = audio_data['features']

                    # Predict vertices
                    vertices = self.predict_vertices(audio_features)

                    if vertices is not None:
                        # Render face
                        rendered_frame = self.renderer.render_vertices(vertices, self.subject)

                        # Calculate latency
                        total_latency = time.time() - audio_data['timestamp']
                        self.latency_buffer.append(total_latency)
                else:
                    # No audio data - show placeholder
                    cv2.putText(
                        rendered_frame,
                        "Waiting for audio...",
                        (150, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )

                # Combine frames side by side
                combined_frame = np.hstack([webcam_frame, rendered_frame])

                # Add performance info
                fps = len(self.fps_counter) / max(sum(self.fps_counter), 0.001)
                avg_latency = np.mean(self.latency_buffer) if self.latency_buffer else 0

                info_text = f"FPS: {fps:.1f} | Latency: {avg_latency*1000:.0f}ms"  # NOQA E226
                cv2.putText(combined_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Recording indicator
                if recording:
                    cv2.circle(combined_frame, (1250, 30), 10, (0, 0, 255), -1)
                    cv2.putText(combined_frame, "REC", (1200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Save frame if recording
                if recording and video_writer:
                    video_writer.write(combined_frame)

                # Display
                cv2.imshow(window_name, combined_frame)

                # Handle key events
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Toggle recording
                    if not recording:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        output_path = f"realtime_recording_{timestamp}.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (1280, 480))
                        recording = True
                        logging.info(f"Started recording: {output_path}")
                    else:
                        recording = False
                        if video_writer:
                            video_writer.release()
                            video_writer = None
                        logging.info("Stopped recording")

                # Update FPS counter
                frame_time = time.time() - frame_start
                self.fps_counter.append(frame_time)

                # Frame rate limiting (optional)
                target_fps = 60
                target_frame_time = 1.0 / target_fps
                if frame_time < target_frame_time:
                    time.sleep(target_frame_time - frame_time)

        except KeyboardInterrupt:
            logging.info("Interrupted by user")
        except Exception as e:
            logging.error(f"Runtime error: {e}")
        finally:
            self.cleanup()
            if recording and video_writer:
                video_writer.release()

    def cleanup(self):
        """Cleanup all resources."""
        self.running = False

        if self.audio_processor:
            self.audio_processor.cleanup()

        if self.renderer:
            self.renderer.cleanup()

        if self.webcam:
            self.webcam.release()

        cv2.destroyAllWindows()
        logging.info("Cleanup completed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Real-Time Face Animation Application")
    parser.add_argument("--model_name", type=str, default="FaceXHuBERT", help="Model name")
    parser.add_argument("--dataset_type", type=str, choices=["BIWI", "VOCASET"], default="BIWI", help="Dataset type")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--feature_dim", type=int, default=256, help="GRU hidden size")
    parser.add_argument("--input_fps", type=int, default=50, help="HuBERT feature extraction fps")
    parser.add_argument("--output_fps", type=int, default=None, help="Target output fps (auto-set)")
    parser.add_argument("--train_subjects", type=str, default="", help="Training subjects (auto-set)")

    args = parser.parse_args()

    # Add dataset attribute for backward compatibility
    args.dataset = args.dataset_type

    # Auto-configure based on dataset
    args = auto_configure_args(args)

    # Create and run application
    app = RealTimeFaceApp(args)

    print("\n" + "=" * 60)
    print("Real-Time Face Animation Application")
    print("=" * 60)
    print(f"Dataset: {args.dataset_type}")
    print(f"Device: {args.device}")
    print("Subject: First training subject (auto-selected)")
    print("\nControls:")
    print("  'q' - Quit application")
    print("  'r' - Toggle recording")
    print("\nStarting application...")
    print("=" * 60 + "\n")

    app.run()


if __name__ == "__main__":
    main()
