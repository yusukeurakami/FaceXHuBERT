"""
Dataset configuration management for FaceXHuBERT.
Centralizes all dataset-specific parameters and logic.
"""

import os
import re

DATASET_CONFIGS = {
    'BIWI': {
        'vertice_dim': 70110,
        'num_vertices': 23370,
        'fps': 25,
        'template_file': 'templates_scaled.pkl',
        'template_type': 'pickle',
        'topology_file': 'templates/BIWI_topology.obj',
        'default_subjects': "F1 F2 F3 F4 F5 F6 F7 F8 M1 M2 M3 M4 M5 M6",
        'coordinate_transform': True,  # Apply coordinate transformation
        'splits': {
            'train': list(range(1, 37)) + list(range(41, 77)),
            'val': list(range(37, 39)) + list(range(77, 79)),
            'test': list(range(39, 41)) + list(range(79, 81)),
        },
        'emotion_threshold': 40,
        'patterns': {'subject': r'([FM]\d+)_.*', 'sentence': r'.*_(\w\d+)\..*'},
    },
    'VOCASET': {
        'vertice_dim': 15069,
        'num_vertices': 5023,
        'fps': 60,
        'template_file': 'templates/templates.pkl',
        'template_type': 'pickle',
        'topology_file': 'templates/FLAME_sample.ply',
        'default_subjects': "FaceTalk_170725_00137 FaceTalk_170904_03276 FaceTalk_170915_00223 FaceTalk_170913_03279 FaceTalk_170728_03272",
        'coordinate_transform': True,  # Use direct FLAME coordinates
        'splits': {
            'train': list(range(1, 33)),
            'val': list(range(33, 37)),
            'test': list(range(37, 41)),
        },
        'emotion_threshold': None,  # No emotion distinction
        'patterns': {'subject': r'(FaceTalk_\d+_\d+)_TA_.*', 'sentence': r'.*_sentence(\d+)\..*'},
    },
}


def get_dataset_config(dataset_type):
    """Get configuration for specified dataset type."""
    if dataset_type not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. Supported types: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[dataset_type]


def auto_configure_args(args):
    """Auto-configure arguments based on dataset choice."""
    config = get_dataset_config(args.dataset)

    # Set default values if not specified
    if not hasattr(args, 'vertice_dim') or args.vertice_dim is None:
        args.vertice_dim = config['vertice_dim']
    if not hasattr(args, 'output_fps') or args.output_fps is None:
        args.output_fps = config['fps']
    if not hasattr(args, 'train_subjects') or not args.train_subjects:
        args.train_subjects = config['default_subjects']
    if not hasattr(args, 'val_subjects') or not args.val_subjects:
        args.val_subjects = config['default_subjects']
    if not hasattr(args, 'test_subjects') or not args.test_subjects:
        args.test_subjects = config['default_subjects']

    return args


def parse_filename(filename, dataset_type):
    """Parse filename to extract subject and sentence information."""
    config = get_dataset_config(dataset_type)

    if dataset_type == "BIWI":
        # F1_e01.wav -> subject="F1", sentence=41 (e01+40)
        subject_match = re.match(config['patterns']['subject'], filename)
        sentence_match = re.match(config['patterns']['sentence'], filename)

        if not subject_match or not sentence_match:
            raise ValueError(f"Could not parse BIWI filename: {filename}")

        subject = subject_match.group(1)
        sentence_part = sentence_match.group(1)

        if sentence_part.startswith('e'):
            sentence_id = int(sentence_part[1:]) + 40
        else:
            sentence_id = int(sentence_part)

    elif dataset_type == "VOCASET":
        # FaceTalk_170725_00137_TA_sentence01.wav -> subject="FaceTalk_170725_00137", sentence=1
        subject_match = re.match(config['patterns']['subject'], filename)
        sentence_match = re.match(config['patterns']['sentence'], filename)

        if not subject_match or not sentence_match:
            raise ValueError(f"Could not parse VOCASET filename: {filename}")

        subject = subject_match.group(1)
        sentence_id = int(sentence_match.group(1))
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    return subject, sentence_id


def get_emotion_label(sentence_id, dataset_type):
    """Get emotion label based on sentence ID and dataset type."""
    config = get_dataset_config(dataset_type)

    if config['emotion_threshold'] is None:
        return "neutral"  # VOCASET has no emotion distinction

    return "emotional" if sentence_id > config['emotion_threshold'] else "neutral"


def should_apply_coordinate_transform(dataset_type):
    """Check if coordinate transformation should be applied for this dataset."""
    config = get_dataset_config(dataset_type)
    return config['coordinate_transform']


def validate_dataset_structure(dataset_path, dataset_type):
    """Validate that the dataset directory has the expected structure."""
    config = get_dataset_config(dataset_type)

    # Check if dataset directory exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    # Check required subdirectories
    required_dirs = ['wav', 'vertices_npy']
    if config['template_type'] == 'pickle':
        required_files = [config['template_file']]
    else:
        required_dirs.append('templates')
        required_files = []

    for dir_name in required_dirs:
        dir_path = os.path.join(dataset_path, dir_name)
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Required directory not found: {dir_path}")

    for file_name in required_files:
        file_path = os.path.join(dataset_path, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")

    return True


def get_template_paths(dataset_path, dataset_type):
    """Get template and topology file paths for the dataset."""
    config = get_dataset_config(dataset_type)

    template_path = os.path.join(dataset_path, config['template_file'])
    topology_path = os.path.join(dataset_path, config['topology_file'])

    return template_path, topology_path
