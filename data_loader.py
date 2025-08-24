import os
import pickle
from collections import defaultdict

import librosa
import numpy as np
import torch
import trimesh
from torch.utils import data
from tqdm import tqdm
from transformers import Wav2Vec2Processor

from dataset_config import get_dataset_config, get_emotion_label, parse_filename
from gt_renderer import transform_gt_to_template_space


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, subjects_dict, data_type="train", dataset_type="BIWI"):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.dataset_type = dataset_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))
        self.emo_one_hot_labels = np.eye(2)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        vertice = self.data[index]["vertice"]
        template = self.data[index]["template"]

        # Use dataset-aware parsing
        try:
            subject, sentence_id = parse_filename(file_name, self.dataset_type)
        except ValueError:
            # Fallback to original parsing for backward compatibility
            sentence_part = file_name.split(".")[0].split("_")[1]
            if sentence_part.startswith('e'):
                sentence_id = int(sentence_part[1:])
            else:
                sentence_id = int(sentence_part)
            subject = "_".join(file_name.split("_")[:-1])

        if self.data_type == "train":
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
        else:
            one_hot = self.one_hot_labels

        # Dataset-aware emotion detection
        emotion_label = get_emotion_label(sentence_id, self.dataset_type)
        if emotion_label == "emotional":
            emo_one_hot = self.emo_one_hot_labels[1]
        else:
            emo_one_hot = self.emo_one_hot_labels[0]

        return (
            torch.FloatTensor(audio),
            vertice,
            torch.FloatTensor(template),
            torch.FloatTensor(one_hot),
            file_name,
            torch.FloatTensor(emo_one_hot),
        )

    def __len__(self):
        return self.len


def load_templates(args):
    """Load templates based on dataset type."""
    config = get_dataset_config(args.dataset)
    template_file = os.path.join(args.dataset, config['template_file'])

    if config['template_type'] == 'pickle':
        # BIWI, VOCASET: Load pickle with multiple subject templates
        with open(template_file, 'rb') as fin:
            templates = pickle.load(fin, encoding='latin1')
            # Ensure templates are properly flattened for compatibility
            flattened_templates = {}
            for subject, template in templates.items():
                if len(template.shape) > 1:
                    flattened_templates[subject] = template.flatten()
                else:
                    flattened_templates[subject] = template
            return flattened_templates
    elif config['template_type'] == 'ply':
        # Load single FLAME template for all subjects
        mesh = trimesh.load_mesh(template_file)
        # Create single template for all subjects
        template_dict = {}
        for subject in config['default_subjects'].split():
            template_dict[subject] = mesh.vertices.flatten()
        return template_dict
    else:
        raise ValueError(f"Unsupported template type: {config['template_type']}")


def load_topology(args):
    config = get_dataset_config(args.dataset)
    topology_file = os.path.join(args.dataset, config['topology_file'])
    return trimesh.load_mesh(topology_file, process=False)


def read_data(args):
    print(f"Loading {args.dataset} data...")
    config = get_dataset_config(args.dataset)
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    audio_path = os.path.join(args.dataset, args.wav_path)
    vertices_path = os.path.join(args.dataset, args.vertices_path)
    ckpt = "facebook/hubert-xlarge-ls960-ft"
    processor = Wav2Vec2Processor.from_pretrained(ckpt)

    # Load templates using the new function
    templates = load_templates(args)
    topology = load_topology(args)

    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(fs, desc=f"Processing {args.dataset} audio files"):
            if f.endswith("wav"):
                try:
                    # Parse filename to get subject and sentence info
                    subject_id, sentence_id = parse_filename(f, args.dataset)
                except ValueError as e:
                    print(f"Warning: Could not parse filename {f}: {e}")
                    continue

                wav_path = os.path.join(r, f)
                speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                input_values = processor(
                    speech_array, return_tensors="pt", padding="longest", sampling_rate=sampling_rate
                ).input_values

                key = f.replace("wav", "npy")
                data[key]["audio"] = input_values

                # Get template for this subject
                if subject_id in templates:
                    temp = templates[subject_id]
                else:
                    # print(f"Loaded template for {subject_id} from {templates.keys()} in {args.dataset}")
                    # For VOCASET, try with _TA suffix
                    subject_with_ta = subject_id + "_TA"
                    if subject_with_ta in templates:
                        temp = templates[subject_with_ta]
                    elif args.dataset == "VOCASET" and "default" in templates:
                        temp = templates["default"]
                    else:
                        print(
                            f"Warning: Template not found for subject {subject_id} (tried {subject_with_ta}), skipping {f}"
                        )
                        continue
                if args.dataset == "VOCASET":
                    # flip the template for VOCASET
                    temp = temp.reshape(-1, 3)
                    temp = transform_gt_to_template_space(temp, topology.vertices)
                    temp = temp.flatten()

                data[key]["name"] = f
                data[key]["template"] = temp.reshape((-1))
                data[key]["subject"] = subject_id
                data[key]["sentence_id"] = sentence_id

                vertice_path = os.path.join(vertices_path, f.replace("wav", "npy"))
                if not os.path.exists(vertice_path):
                    # print(f"Vertices Data Not Found: {vertice_path}")
                    continue
                else:
                    data[key]["vertice"] = vertice_path

    subjects_dict = {}
    subjects_dict["train"] = [i for i in args.train_subjects.split(" ")]
    subjects_dict["val"] = [i for i in args.val_subjects.split(" ")]
    subjects_dict["test"] = [i for i in args.test_subjects.split(" ")]

    # Use dataset-specific splits
    splits = config['splits']

    for k, v in data.items():
        subject_id = v.get("subject")
        sentence_id = v.get("sentence_id")

        if subject_id is None or sentence_id is None:
            continue

        # Check if subject is in the training subjects and sentence is in the appropriate split
        if subject_id in subjects_dict["train"] and sentence_id in splits['train']:
            train_data.append(v)
        if subject_id in subjects_dict["val"] and sentence_id in splits['val']:
            valid_data.append(v)
        if subject_id in subjects_dict["test"] and sentence_id in splits['test']:
            test_data.append(v)

    print(f"Loaded {args.dataset} data: train: {len(train_data)}, valid: {len(valid_data)}, test: {len(test_data)}")
    return train_data, valid_data, test_data, subjects_dict


def get_dataloaders(args):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args)
    train_data = Dataset(train_data, subjects_dict, "train", args.dataset)
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    valid_data = Dataset(valid_data, subjects_dict, "val", args.dataset)
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False)
    test_data = Dataset(test_data, subjects_dict, "test", args.dataset)
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    return dataset


if __name__ == "__main__":
    get_dataloaders()
