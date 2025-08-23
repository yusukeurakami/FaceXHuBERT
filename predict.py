import argparse
import os
import pickle as pkl
import time

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2Processor

from dataset_config import auto_configure_args, get_dataset_config
from faceXhubert import FaceXHuBERT
from video_utils import create_video_from_prediction


def test_model(args):
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    model = FaceXHuBERT(args)
    model.load_state_dict(torch.load('pretrained_model/{}.pth'.format(args.model_name)))
    model = model.to(torch.device(args.device))
    model.eval()

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
                # Create mappings for subjects without _TA suffix
                vocaset_templates = {}
                for key, template in templates.items():
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
        templates[args.subject + "_TA"] = mesh.vertices.flatten()
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

    prediction = model.predict(audio_feature, template, one_hot, emo_one_hot)
    prediction = prediction.squeeze()
    elapsed = time.time() - start_time
    print("Inference time for ", prediction.shape[0], " frames is: ", elapsed, " seconds.")
    print("Inference time for 1 frame is: ", elapsed / prediction.shape[0], " seconds.")
    print("Inference time for 1 second of audio is: ", ((elapsed * 25) / prediction.shape[0]), " seconds.")
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
    args = parser.parse_args()

    # Add dataset attribute for backward compatibility
    args.dataset = args.dataset_type
    # Auto-configure arguments based on dataset choice
    args = auto_configure_args(args)

    test_model(args)
    render(args)


if __name__ == "__main__":
    main()
