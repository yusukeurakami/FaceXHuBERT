import gc
import os
import pickle as pkl

import cv2
import ffmpeg
import numpy as np
import pyrender
import trimesh

from gt_renderer import transform_gt_to_template_space


def main():
    train_folder = "render_folder/"
    results_folder = "result/"
    audio_folder = "BIWI/wav/"
    video_woA_folder = "renders/" + train_folder + "videos_no_audio/"
    video_wA_folder = "renders/" + train_folder + "videos_with_audio/"
    frames_folder = "renders/" + train_folder + "temp/frames/"

    # Create folders if they don't exist
    os.makedirs(video_woA_folder, exist_ok=True)
    os.makedirs(video_wA_folder, exist_ok=True)
    os.makedirs(frames_folder, exist_ok=True)

    seqs = os.listdir(results_folder)

    fps = 25
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
    camera_pose = np.array([[1.0, 0, 0.0, 0.00], [0.0, -1.0, 0.0, 0.00], [0.0, 0.0, 1.0, -1.6], [0.0, 0.0, 0.0, 1.0]])

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)

    # r = pyrender.OffscreenRenderer(640, 480)
    r = pyrender.OffscreenRenderer(1920, 1440)

    template_data = None
    with open("BIWI/templates_scaled.pkl", "rb") as f:
        template_data = pkl.load(f)

    # Load the face topology from the reference OBJ file
    topology_mesh = trimesh.load_mesh("BIWI/templates/BIWI_topology.obj", process=False)

    for seq in seqs:
        if seq.endswith('.npy'):
            video_woA_path = video_woA_folder + seq.split('.')[0] + '.mp4'
            video_wA_path = video_wA_folder + seq.split('.')[0] + '.mp4'
            # video = cv2.VideoWriter(video_woA_path, fourcc, fps, (640, 480))
            video = cv2.VideoWriter(video_woA_path, fourcc, fps, (1920, 1440))

            seq_path = results_folder + seq
            subject = seq.split('_')[0]
            audio = seq.split('_')[0] + '_' + seq.split('_')[1] + '.wav'
            audio_path = audio_folder + audio

            # Create mesh using template vertices and topology faces
            ref_mesh = trimesh.Trimesh(vertices=template_data[subject], faces=topology_mesh.faces)

            seq = np.load(seq_path)
            seq = np.reshape(seq, (-1, 70110 // 3, 3))

            seq_transformed = np.zeros_like(seq)
            for f in range(seq.shape[0]):
                seq_transformed[f] = transform_gt_to_template_space(seq[f], template_data[subject])

            ref_mesh.vertices = seq_transformed[0, :, :]
            py_mesh = pyrender.Mesh.from_trimesh(ref_mesh)
            for f in range(seq.shape[0]):
                ref_mesh.vertices = seq_transformed[f, :, :]
                py_mesh = pyrender.Mesh.from_trimesh(ref_mesh)
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

            input_video = ffmpeg.input(video_woA_path)
            input_audio = ffmpeg.input(audio_path)
            ffmpeg.concat(input_video, input_audio, v=1, a=1).output(video_wA_path).run()
            del video, seq, ref_mesh
            gc.collect()


if __name__ == "__main__":
    main()
