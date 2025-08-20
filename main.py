import argparse
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data_loader import get_dataloaders
from dataset_config import auto_configure_args, should_apply_coordinate_transform
from faceXhubert import FaceXHuBERT
from gt_renderer import transform_gt_to_template_space
from video_utils import create_video_from_prediction


def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.legend()
    plt.title("Losses")
    plt.savefig("losses.png")
    plt.close()


def trainer(args, train_loader, dev_loader, model, optimizer, criterion, epoch=100):
    train_losses = []
    val_losses = []

    save_path = os.path.join(args.save_path)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    train_subjects_list = [i for i in args.train_subjects.split(" ")]
    iteration = 0
    for e in range(epoch + 1):
        loss_log = []
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        optimizer.zero_grad()

        for i, (audio, vertice, template, one_hot, file_name, emo_one_hot) in pbar:
            iteration += 1
            vertice = str(vertice[0])
            vertice = np.load(vertice, allow_pickle=True)
            vertice = vertice.astype(np.float32)

            # Apply transformation only for datasets that require it (BIWI)
            if should_apply_coordinate_transform(args.dataset):
                # Apply transformation to match template coordinate system
                template_vertices = template[0].cpu().numpy().reshape(-1, 3)  # Reshape to (num_vertices, 3)

                # Reshape vertice data for transformation
                num_frames, total_coords = vertice.shape
                num_vertices = total_coords // 3
                vertice_reshaped = vertice.reshape(num_frames, num_vertices, 3)

                # Apply transformation frame by frame
                transformed_frames = []
                for frame_idx in range(num_frames):
                    transformed_frame = transform_gt_to_template_space(vertice_reshaped[frame_idx], template_vertices)
                    transformed_frames.append(transformed_frame)

                # Reshape back to original format
                vertice = np.array(transformed_frames).reshape(num_frames, -1)

            vertice = torch.from_numpy(vertice)
            vertice = torch.unsqueeze(vertice, 0)
            audio, vertice, template, one_hot, emo_one_hot = (
                audio.to(device="cuda"),
                vertice.to(device="cuda"),
                template.to(device="cuda"),
                one_hot.to(device="cuda"),
                emo_one_hot.to(device="cuda"),
            )
            loss = model(audio, template, vertice, one_hot, emo_one_hot, criterion)

            loss.backward()
            loss_log.append(loss.item())
            if i % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                del audio, vertice, template, one_hot, emo_one_hot
                torch.cuda.empty_cache()

            pbar.set_description(
                "(Epoch {}, iteration {}) TRAIN LOSS:{:.8f}".format((e + 1), iteration, np.mean(loss_log))
            )

        train_losses.append(np.mean(loss_log))

        valid_loss_log = []
        model.eval()
        for audio, vertice, template, one_hot_all, file_name, emo_one_hot in dev_loader:
            # to gpu
            vertice = str(vertice[0])
            vertice = np.load(vertice, allow_pickle=True)
            vertice = vertice.astype(np.float32)

            # Apply transformation only for datasets that require it (BIWI)
            if should_apply_coordinate_transform(args.dataset):
                # Apply transformation to match template coordinate system
                template_vertices = template[0].cpu().numpy().reshape(-1, 3)  # Reshape to (num_vertices, 3)

                # Reshape vertice data for transformation
                num_frames, total_coords = vertice.shape
                num_vertices = total_coords // 3
                vertice_reshaped = vertice.reshape(num_frames, num_vertices, 3)

                # Apply transformation frame by frame
                transformed_frames = []
                for frame_idx in range(num_frames):
                    transformed_frame = transform_gt_to_template_space(vertice_reshaped[frame_idx], template_vertices)
                    transformed_frames.append(transformed_frame)

                # Reshape back to original format
                vertice = np.array(transformed_frames).reshape(num_frames, -1)

            vertice = torch.from_numpy(vertice)
            vertice = torch.unsqueeze(vertice, 0)
            audio, vertice, template, one_hot_all, emo_one_hot = (
                audio.to(device="cuda"),
                vertice.to(device="cuda"),
                template.to(device="cuda"),
                one_hot_all.to(device="cuda"),
                emo_one_hot.to(device="cuda"),
            )
            train_subject = "_".join(file_name[0].split("_")[:-1])
            if train_subject in train_subjects_list:
                condition_subject = train_subject
                iter = train_subjects_list.index(condition_subject)
                one_hot = one_hot_all[:, iter, :]
                loss = model(audio, template, vertice, one_hot, emo_one_hot, criterion)
                valid_loss_log.append(loss.item())
            else:
                for iter in range(one_hot_all.shape[-1]):
                    condition_subject = train_subjects_list[iter]
                    one_hot = one_hot_all[:, iter, :]
                    loss = model(audio, template, vertice, one_hot, emo_one_hot, criterion)
                    valid_loss_log.append(loss.item())

        current_loss = np.mean(valid_loss_log)

        val_losses.append(current_loss)
        if (e > 0 and e % 25 == 0) or e == args.max_epoch:
            torch.save(model.state_dict(), os.path.join(save_path, '{}_model.pth'.format(e)))

        print("epcoh: {}, current loss:{:.8f}".format(e + 1, current_loss))

        plot_losses(train_losses, val_losses)

    return model


@torch.no_grad()
def test(args, model, test_loader, epoch):
    result_path = os.path.join(args.result_path)
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)

    # Create video output directory
    video_output_dir = os.path.join(args.result_path, "videos")
    os.makedirs(video_output_dir, exist_ok=True)

    save_path = os.path.join(args.save_path)
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    model.load_state_dict(torch.load(os.path.join(save_path, '{}_model.pth'.format(epoch))))
    model = model.to(torch.device("cuda"))
    model.eval()

    for audio, vertice, template, one_hot_all, file_name, emo_one_hot in test_loader:
        vertice = str(vertice[0])
        vertice = np.load(vertice, allow_pickle=True)
        vertice = vertice.astype(np.float32)

        # Apply transformation only for datasets that require it (BIWI)
        if should_apply_coordinate_transform(args.dataset):
            # Apply transformation to match template coordinate system
            template_vertices = template[0].cpu().numpy().reshape(-1, 3)  # Reshape to (num_vertices, 3)

            # Reshape vertice data for transformation
            num_frames, total_coords = vertice.shape
            num_vertices = total_coords // 3
            vertice_reshaped = vertice.reshape(num_frames, num_vertices, 3)

            # Apply transformation frame by frame
            transformed_frames = []
            for frame_idx in range(num_frames):
                transformed_frame = transform_gt_to_template_space(vertice_reshaped[frame_idx], template_vertices)
                transformed_frames.append(transformed_frame)

            # Reshape back to original format
            vertice = np.array(transformed_frames).reshape(num_frames, -1)

        vertice = torch.from_numpy(vertice)
        vertice = torch.unsqueeze(vertice, 0)
        audio, vertice, template, one_hot_all, emo_one_hot = (
            audio.to(device="cuda"),
            vertice.to(device="cuda"),
            template.to(device="cuda"),
            one_hot_all.to(device="cuda"),
            emo_one_hot.to(device="cuda"),
        )

        # Extract subject information and determine emotion label
        train_subject = "_".join(file_name[0].split("_")[:-1])
        base_name = file_name[0].split(".")[0]

        # Determine emotion label based on sentence ID (following data_loader.py logic)
        sentence_part = file_name[0].split(".")[0].split("_")[1]
        if sentence_part.startswith('e'):
            sentence_id = int(sentence_part[1:])
        else:
            sentence_id = int(sentence_part)
        emotion_label = "emotional" if sentence_id > 40 else "neutral"

        # Get corresponding audio file path for video rendering
        audio_file_path = os.path.join(args.dataset, args.wav_path, file_name[0])

        if train_subject in train_subjects_list:
            condition_subject = train_subject
            iter = train_subjects_list.index(condition_subject)
            one_hot = one_hot_all[:, iter, :]
            prediction = model.predict(audio, template, one_hot, emo_one_hot)
            prediction = prediction.squeeze()

            # Save prediction
            prediction_filename = base_name + "_condition_" + condition_subject + ".npy"
            prediction_path = os.path.join(result_path, prediction_filename)
            np.save(prediction_path, prediction.detach().cpu().numpy())
        else:
            for iter in range(one_hot_all.shape[-1]):
                condition_subject = train_subjects_list[iter]
                one_hot = one_hot_all[:, iter, :]
                prediction = model.predict(audio, template, one_hot, emo_one_hot)
                prediction = prediction.squeeze()

                # Save prediction
                prediction_filename = base_name + "_condition_" + condition_subject + ".npy"
                prediction_path = os.path.join(result_path, prediction_filename)
                np.save(prediction_path, prediction.detach().cpu().numpy())

        # Create video
        try:
            create_video_from_prediction(
                prediction_path=prediction_path,
                subject=train_subject,
                condition_subject=condition_subject,
                base_name=base_name,
                output_dir=video_output_dir,
                emotion_label=emotion_label,
                audio_path=audio_file_path if os.path.exists(audio_file_path) else None,
                fps=args.output_fps,
                dataset_type=args.dataset,
            )
            print(f"Video created for {prediction_filename}")
        except Exception as e:
            print(f"Warning: Could not create video for {prediction_filename}. Error: {e}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(
        description='FaceXHuBERT: Text-less Speech-driven E(X)pressive 3D Facial Animation Synthesis using Self-Supervised Speech Representation Learning'
    )
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument(
        "--dataset", type=str, choices=["BIWI", "VOCASET"], default="BIWI", help='Dataset type: BIWI or VOCASET'
    )
    parser.add_argument(
        "--vertice_dim", type=int, default=None, help='number of vertices (auto-set based on dataset if not specified)'
    )
    parser.add_argument("--feature_dim", type=int, default=256, help='GRU Vertex decoder hidden size')
    parser.add_argument("--wav_path", type=str, default="wav", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="vertices_npy", help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=100, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--template_file",
        type=str,
        default=None,
        help='path of the train subject templates (auto-set based on dataset)',
    )
    parser.add_argument("--save_path", type=str, default="save", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    parser.add_argument(
        "--train_subjects", type=str, default="", help='training subjects (auto-set based on dataset if not specified)'
    )
    parser.add_argument(
        "--val_subjects", type=str, default="", help='validation subjects (auto-set based on dataset if not specified)'
    )
    parser.add_argument(
        "--test_subjects", type=str, default="", help='test subjects (auto-set based on dataset if not specified)'
    )
    parser.add_argument(
        "--input_fps", type=int, default=50, help='HuBERT last hidden state produces 50 fps audio representation'
    )
    parser.add_argument(
        "--output_fps",
        type=int,
        default=None,
        help='fps of the visual data (auto-set based on dataset if not specified)',
    )
    args = parser.parse_args()

    # Auto-configure arguments based on dataset choice
    args = auto_configure_args(args)

    model = FaceXHuBERT(args)
    print("model parameters: ", count_parameters(model))

    assert torch.cuda.is_available()

    model = model.to(torch.device("cuda"))
    dataset = get_dataloaders(args)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    model = trainer(args, dataset["train"], dataset["valid"], model, optimizer, criterion, epoch=args.max_epoch)

    test(args, model, dataset["test"], epoch=args.max_epoch)

    print(model)


if __name__ == "__main__":
    main()
