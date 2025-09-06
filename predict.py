import argparse
import os
import pickle as pkl
import time

import librosa
import numpy as np
import torch
import trimesh
from transformers import Wav2Vec2Processor

from dataset_config import auto_configure_args, get_dataset_config
from faceXhubert import FaceXHuBERT
from video_utils import create_video_from_prediction, transform_gt_to_template_space


def load_topology(args):
    config = get_dataset_config(args.dataset)
    topology_file = os.path.join(args.dataset, config['topology_file'])
    return trimesh.load_mesh(topology_file, process=False)


def test_model(args):
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    model = FaceXHuBERT(args)
    model.load_state_dict(torch.load('pretrained_model/{}.pth'.format(args.model_name)))
    model = model.to(torch.device(args.device))
    model.eval()
    print("Model architecture:\n", model)
    print("Model loaded from: pretrained_model/{}.pth".format(args.model_name))
    print("Model device:", args.device)

    # Print parameter size and memory consumption for major modules
    def sizeof_fmt(num, suffix='B'):
        for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
            if abs(num) < 1024.0:
                return "%3.1f%s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f%s%s" % (num, 'Y', suffix)

    total_params = 0
    print("\nParameter size and memory consumption by major modules:")
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        module_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
        total_params += module_params
        print(f"  {name}: {module_params:,} params, {sizeof_fmt(module_bytes)}")

    # Print total
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Total memory (parameters): {sizeof_fmt(total_bytes)}")

    # Load templates based on dataset type
    config = get_dataset_config(args.dataset_type)
    template_file = os.path.join(args.dataset_type, config['template_file'])

    if config['template_type'] == 'pickle':
        with open(template_file, 'rb') as fin:
            templates = pkl.load(fin, encoding='latin1')
            # Flatten templates if needed
            flattened_templates = {}
            for subject, template in templates.items():
                if len(template.shape) > 1:
                    flattened_templates[subject] = template.flatten()
                else:
                    flattened_templates[subject] = template
            templates = flattened_templates

            # For VOCASET, handle _TA suffix in template keys
            if args.dataset_type == "VOCASET":
                topology = load_topology(args)
                # Create mappings for subjects without _TA suffix
                vocaset_templates = {}
                for key, template in templates.items():

                    # 20250903
                    # flip the template for VOCASET
                    template = template.reshape(-1, 3)
                    template = transform_gt_to_template_space(template, topology.vertices)
                    template = template.flatten()

                    if key.endswith('_TA'):
                        # Map both with and without _TA suffix
                        base_key = key[:-3]  # Remove _TA
                        vocaset_templates[base_key] = template
                        vocaset_templates[key] = template
                    else:
                        vocaset_templates[key] = template
                templates = vocaset_templates
    elif config['template_type'] == 'ply':
        import trimesh

        mesh = trimesh.load_mesh(template_file)
        # Create template for the subject (try both with and without _TA suffix)
        templates = {args.subject: mesh.vertices.flatten()}
        templates[args.subject] = mesh.vertices.flatten()
    else:
        raise ValueError(f"Unsupported template type: {config['template_type']}")

    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    one_hot_labels = np.eye(len(train_subjects_list))
    emo_one_hot_labels = np.eye(2)
    if args.emotion == 1:
        emo_one_hot = torch.FloatTensor(emo_one_hot_labels[1]).to(device=args.device)
        emo_label = "emotional"
    else:
        emo_one_hot = torch.FloatTensor(emo_one_hot_labels[0]).to(device=args.device)
        emo_label = "neutral"

    iter = train_subjects_list.index(args.condition)
    one_hot = one_hot_labels[iter]
    one_hot = np.reshape(one_hot, (-1, one_hot.shape[0]))
    one_hot = torch.FloatTensor(one_hot).to(device=args.device)

    temp = templates[args.subject]

    template = temp.reshape((-1))
    template = np.reshape(template, (-1, template.shape[0]))
    template = torch.FloatTensor(template).to(device=args.device)

    # Check CUDA memory before HuBERT loading
    if torch.cuda.is_available():
        print(f"CUDA memory before HuBERT: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    wav_path = args.wav_path
    test_name = os.path.basename(wav_path).split(".")[0]
    start_time = time.time()
    speech_array, sampling_rate = librosa.load(os.path.join(wav_path), sr=16000)
    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-xlarge-ls960-ft")
    audio_feature = processor(
        speech_array, return_tensors="pt", padding="longest", sampling_rate=sampling_rate
    ).input_values
    audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
    audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)

    # After HuBERT loading (around line 135), add:
    if torch.cuda.is_available():
        print(f"CUDA memory after HuBERT: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"CUDA reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

    # Debug input data sizes
    print("\nInput data sizes:")
    print(
        f"speech_array shape: {speech_array.shape}, dtype: {speech_array.dtype}, size: {speech_array.nbytes / 1024**2:.2f} MB"
    )
    print(f"audio_feature shape: {audio_feature.shape}, dtype: {audio_feature.dtype}")
    print(f"audio_feature size: {audio_feature.numel() * audio_feature.element_size() / 1024**2:.2f} MB")
    print(f"template shape: {template.shape}, dtype: {template.dtype}")
    print(f"template size: {template.numel() * template.element_size() / 1024**2:.2f} MB")
    print(f"one_hot shape: {one_hot.shape}, dtype: {one_hot.dtype}")
    print(f"one_hot size: {one_hot.numel() * one_hot.element_size() / 1024**2:.2f} MB")
    print(f"emo_one_hot shape: {emo_one_hot.shape}, dtype: {emo_one_hot.dtype}")
    print(f"emo_one_hot size: {emo_one_hot.numel() * emo_one_hot.element_size() / 1024**2:.2f} MB")

    # Check total input data size
    total_input_size = (
        audio_feature.numel() * audio_feature.element_size()
        + template.numel() * template.element_size()
        + one_hot.numel() * one_hot.element_size()
        + emo_one_hot.numel() * emo_one_hot.element_size()
    ) / 1024**2
    print(f"Total input data size: {total_input_size:.2f} MB")

    with torch.no_grad():
        print(f"CUDA memory before prediction: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        prediction = model.predict(audio_feature, template, one_hot, emo_one_hot)
        print(f"CUDA memory after prediction: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        prediction = prediction.squeeze()
        print(f"CUDA memory after squeeze: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"Prediction shape: {prediction.shape}, dtype: {prediction.dtype}")
        print(f"Prediction size: {prediction.numel() * prediction.element_size() / 1024**2:.2f} MB")

    elapsed = time.time() - start_time
    if torch.cuda.is_available():
        print(f"CUDA memory after prediction: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"CUDA reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
    print("Inference time for ", prediction.shape[0], " frames is: ", elapsed, " seconds.")
    print("Inference time for 1 frame is: ", elapsed / prediction.shape[0], " seconds.")
    print("Inference time for 1 second of audio is: ", ((elapsed * 25) / prediction.shape[0]), " seconds.")
    print("Inference frequency: {:.2f} Hz".format(prediction.shape[0] / elapsed if elapsed > 0 else 0))
    out_file_name = test_name + "_" + emo_label + "_" + args.subject + "_Condition_" + args.condition
    np.save(os.path.join(args.result_path, out_file_name), prediction.detach().cpu().numpy())


def render(args):
    emo_label = "emotional" if args.emotion == 1 else "neutral"
    wav_path = args.wav_path
    test_name = os.path.basename(wav_path).split(".")[0]
    out_file_name = test_name + "_" + emo_label + "_" + args.subject + "_Condition_" + args.condition
    predicted_vertices_path = os.path.join(args.result_path, out_file_name + ".npy")

    print("Rendering the predicted sequence:", test_name)

    # Use the shared video utils for rendering with dataset type
    create_video_from_prediction(
        prediction_path=predicted_vertices_path,
        subject=args.subject,
        condition_subject=args.condition,
        base_name=test_name,
        output_dir="demo/render",
        emotion_label=emo_label,
        audio_path=wav_path,
        fps=args.fps,
        dataset_type=args.dataset_type,
        zoom_factor=args.zoom_factor,
        camera_distance=args.camera_distance,
        apply_transform=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description='FaceXHuBERT: Text-less Speech-driven E(X)pressive 3D Facial Animation Synthesis using Self-Supervised Speech Representation Learning'
    )
    parser.add_argument("--model_name", type=str, default="FaceXHuBERT")
    parser.add_argument(
        "--dataset_type", type=str, choices=["BIWI", "VOCASET"], default="BIWI", help='Dataset type for prediction'
    )
    parser.add_argument("--fps", type=float, default=None, help='frame rate (auto-set based on dataset)')
    parser.add_argument("--feature_dim", type=int, default=256, help='GRU Vertex Decoder hidden size')
    parser.add_argument("--vertice_dim", type=int, default=None, help='number of vertices (auto-set based on dataset)')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--train_subjects", type=str, default="", help='training subjects (auto-set based on dataset if not specified)'
    )
    parser.add_argument(
        "--test_subjects", type=str, default="", help='test subjects (auto-set based on dataset if not specified)'
    )
    parser.add_argument(
        "--wav_path", type=str, default="demo/wav/test.wav", help='path of the input audio signal in .wav format'
    )
    parser.add_argument("--result_path", type=str, default="demo/result", help='path of the predictions in .npy format')
    parser.add_argument("--condition", type=str, default="M3", help='select a conditioning subject from train_subjects')
    parser.add_argument(
        "--subject", type=str, default="M1", help='select a subject from test_subjects or train_subjects'
    )
    parser.add_argument(
        "--template_path", type=str, default=None, help='path of the personalized templates (auto-set based on dataset)'
    )
    parser.add_argument(
        "--render_template_path", type=str, default="templates", help='path of the mesh in BIWI topology'
    )
    parser.add_argument(
        "--input_fps", type=int, default=50, help='HuBERT last hidden state produces 50 fps audio representation'
    )
    parser.add_argument(
        "--output_fps", type=int, default=None, help='fps of the visual data (auto-set based on dataset)'
    )
    parser.add_argument(
        "--emotion",
        type=int,
        default="1",
        help='style control for emotion, 1 for expressive animation, 0 for neutral animation',
    )
    parser.add_argument(
        "--zoom_factor", type=float, default=1.0, help='zoom factor for field of view (>1.0 = zoom in, <1.0 = zoom out)'
    )
    parser.add_argument(
        "--camera_distance", type=float, default=-1.6, help='distance of camera from object (negative values = closer)'
    )
    args = parser.parse_args()

    # Add dataset attribute for backward compatibility
    args.dataset = args.dataset_type
    # Auto-configure arguments based on dataset choice
    args = auto_configure_args(args)

    test_model(args)
    render(args)


if __name__ == "__main__":
    main()
