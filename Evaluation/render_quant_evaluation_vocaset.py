import gc
import os
import pickle as pkl

import cv2
import ffmpeg
import numpy as np
import pyrender
import trimesh

from gt_renderer import transform_gt_to_template_space

quantfilename = "quantitative_metric.txt"
render_folder = "renders/"
gt_folder = "GroundTruth/"
pred_folder = "../result/"
audio_folder = "../VOCASET/wav/"
video_woA_folder = render_folder + "videos_no_audio/"
video_wA_folder = render_folder + "videos_with_audio/"
meshes_folder = render_folder + "temp/meshes/"
frames_folder = render_folder + "temp/frames/"

mean_face_vertex_error = 0

gt_seqs = os.listdir(gt_folder)
pred_seqs = os.listdir(pred_folder)

fps = 60  # VOCASET uses 60 FPS
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
zoom_factor = 3.0

base_fov = np.pi / 3.0
adjusted_fov = base_fov / zoom_factor
cam = pyrender.PerspectiveCamera(yfov=adjusted_fov, aspectRatio=1.414)
camera_pose = np.array([[1.0, 0, 0.0, 0.00], [0.0, -1.0, 0.0, 0.00], [0.0, 0.0, 1.0, -1.6], [0.0, 0.0, 0.0, 1.0]])

light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)

r = pyrender.OffscreenRenderer(1920, 1440)

# Load VOCASET templates
template_data = None
with open("../VOCASET/templates/templates.pkl", "rb") as f:
    template_data = pkl.load(f, encoding='latin1')

# Load the FLAME topology from the VOCASET template
topology_mesh = trimesh.load_mesh("../VOCASET/templates/FLAME_sample.ply", process=False)

print("Evaluation started")
for gt_seq in gt_seqs:
    if gt_seq.endswith('.npy'):
        video_woA_path = video_woA_folder + gt_seq.split('.')[0] + '.mp4'
        video_wA_path = video_wA_folder + gt_seq.split('.')[0] + '.mp4'
        video = cv2.VideoWriter(video_woA_path, fourcc, fps, (1920, 1440))
        gt_seq_path = gt_folder + gt_seq
        print("Now evaluating sequence: ", gt_seq)

        # Parse VOCASET filename to get subject and sentence info
        # Format: FaceTalk_170904_03276_TA_sentence28.npy
        filename_parts = gt_seq.split('.')[0].split('_')
        subject = '_'.join(filename_parts[:3])  # FaceTalk_170904_03276
        sentence_num = filename_parts[-1].replace('sentence', '')  # 28

        # Find the corresponding prediction file
        # Look for files with the pattern: {gt_filename}_condition_{subject}.npy
        gt_base_name = gt_seq.split('.')[0]  # FaceTalk_170904_03276_TA_sentence38
        pred_pattern = f"{gt_base_name}_condition_{subject}.npy"
        pred_seq_path = pred_folder + pred_pattern

        # Construct audio filename
        audio = f"{subject}_TA_sentence{sentence_num}.wav"
        audio_path = audio_folder + audio

        # Get template for this subject (VOCASET templates may have _TA suffix)
        if subject in template_data:
            template_vertices = template_data[subject]
        elif subject + "_TA" in template_data:
            template_vertices = template_data[subject + "_TA"]
        else:
            print(f"Warning: Template not found for subject {subject}, using default")
            # Use first available template as fallback
            template_vertices = list(template_data.values())[0]

        # Create mesh using template vertices and topology faces
        render_mesh = trimesh.Trimesh(vertices=template_vertices, faces=topology_mesh.faces)

        gt_seq_data = np.load(gt_seq_path)
        try:
            pred_seq = np.load(pred_seq_path)
        except Exception as e:
            print(f"Warning: Could not load prediction file: {e}")
            continue

        if gt_seq_data.shape[0] > pred_seq.shape[0]:
            gt_seq_data = gt_seq_data[: pred_seq.shape[0]]

        if pred_seq.shape[0] > gt_seq_data.shape[0]:
            pred_seq = pred_seq[: gt_seq_data.shape[0]]

        # VOCASET has 15,069 dimensions (5,023 vertices * 3)
        gt_seq_data = np.reshape(gt_seq_data, (-1, 15069 // 3, 3))
        pred_seq = np.reshape(pred_seq, (-1, 15069 // 3, 3))
        assert gt_seq_data.shape == pred_seq.shape
        sequence_mean_face_vertex_error = 0

        # GT data needs to be transformed to template space
        gt_seq_transformed = np.zeros_like(gt_seq_data)
        for f in range(gt_seq_data.shape[0]):
            gt_seq_transformed[f] = transform_gt_to_template_space(gt_seq_data[f], template_vertices)

        # Pred data is already in template space
        pred_seq_transformed = pred_seq

        for f in range(pred_seq.shape[0]):

            # Calculate vertex error using numpy instead of pymeshlab for now
            vertex_error = np.linalg.norm(gt_seq_transformed[f] - pred_seq_transformed[f], axis=1)
            sequence_mean_face_vertex_error = sequence_mean_face_vertex_error + np.mean(vertex_error)

            render_mesh.vertices = pred_seq_transformed[f, :, :]
            py_mesh = pyrender.Mesh.from_trimesh(render_mesh)
            scene = pyrender.Scene()
            scene.add(py_mesh)

            scene.add(cam, pose=camera_pose)
            scene.add(light, pose=camera_pose)
            color, _ = r.render(scene)

            output_frame = frames_folder + "frame" + str(f) + ".jpg"
            image_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_frame, image_bgr)
            frame = cv2.imread(output_frame)
            video.write(frame)
        video.release()
        sequence_mean_face_vertex_error = sequence_mean_face_vertex_error / pred_seq.shape[0]
        mean_face_vertex_error = mean_face_vertex_error + sequence_mean_face_vertex_error

        # Add audio to video if audio file exists
        if os.path.exists(audio_path):
            input_video = ffmpeg.input(video_woA_path)
            input_audio = ffmpeg.input(audio_path)
            ffmpeg.concat(input_video, input_audio, v=1, a=1).output(video_wA_path).run()
        else:
            print(f"Warning: Audio file not found: {audio_path}")
            # Copy video without audio
            import shutil

            shutil.copy2(video_woA_path, video_wA_path)

        del video
        gc.collect()

mean_face_vertex_error = mean_face_vertex_error / len(gt_seqs)

file = open(quantfilename, "w")

# convert variable to string
str = repr(mean_face_vertex_error)
file.write("mean_face_vertex_error = " + str + "\n")

file.close()
print("Done!")


def main():
    """Main entry point for quantitative evaluation."""
    pass  # The evaluation logic is executed at module level


if __name__ == "__main__":
    main()
