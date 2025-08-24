from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from hubert.modeling_hubert import HubertModel


def inputRepresentationAdjustment(
    audio_embedding_matrix: torch.Tensor, vertex_matrix: torch.Tensor, ifps: int, ofps: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Adjusts audio embeddings and vertex matrices to align frame rates for facial animation synthesis.

    This function handles temporal alignment between audio features and facial vertex data when
    the input and output frame rates differ. It supports two scenarios:

    1. **Compatible frame rates** (ifps % ofps == 0): Direct frame rate conversion by
       skipping/combining frames based on the conversion factor.
    2. **Incompatible frame rates** (ifps % ofps != 0): Linear interpolation of audio
       embeddings to match the target sequence length.

    The function ensures that audio embeddings and vertex matrices have compatible temporal
    dimensions for the GRU-based facial animation synthesis.

    Args:
        audio_embedding_matrix (torch.Tensor): Audio feature embeddings from HuBERT model.
            Shape: (batch_size, seq_len, audio_dim)
        vertex_matrix (torch.Tensor): Facial vertex coordinates for animation.
            Shape: (batch_size, seq_len, vertex_dim)
        ifps (int): Input frame rate (audio sampling rate)
        ofps (int): Output frame rate (target facial animation frame rate)

    Returns:
        tuple: A tuple containing:
            - audio_embedding_matrix (torch.Tensor): Adjusted audio embeddings with
              compatible temporal dimensions. Shape: (1, adjusted_seq_len, audio_dim * factor)
            - vertex_matrix (torch.Tensor): Potentially truncated vertex matrix to match
              audio embedding length
            - frame_num (int): Final number of frames in the vertex matrix

    Example:
        >>> # For BIWI dataset: 50fps audio → 25fps video
        >>> audio_emb, verts, frames = inputRepresentationAdjustment(
        ...     audio_features, vertex_data, ifps=50, ofps=25)
        >>> # Factor = 2, so 2 audio frames → 1 video frame

        >>> # For VOCASET: 50fps audio → 60fps video
        >>> audio_emb, verts, frames = inputRepresentationAdjustment(
        ...     audio_features, vertex_data, ifps=50, ofps=60)
        >>> # Uses interpolation to create 60fps audio features

    Factor Calculation Examples:
        >>> # If ifps = 30 and ofps = 10, then factor = 3
        >>> # (meaning 3 input frames → 1 output frame)
        >>> # If ifps = 10 and ofps = 30, then factor = 1
        >>> # (meaning 1 input frame → 1 output frame, with interpolation)
    """
    if ifps % ofps == 0:
        factor = -1 * (-ifps // ofps)
        if audio_embedding_matrix.shape[1] % 2 != 0:
            audio_embedding_matrix = audio_embedding_matrix[:, : audio_embedding_matrix.shape[1] - 1]

        if audio_embedding_matrix.shape[1] > vertex_matrix.shape[1] * 2:
            audio_embedding_matrix = audio_embedding_matrix[:, : vertex_matrix.shape[1] * 2]

        elif audio_embedding_matrix.shape[1] < vertex_matrix.shape[1] * 2:
            vertex_matrix = vertex_matrix[:, : audio_embedding_matrix.shape[1] // 2]
    else:
        factor = -1 * (-ifps // ofps)
        audio_embedding_seq_len = vertex_matrix.shape[1] * factor
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)
        audio_embedding_matrix = F.interpolate(
            audio_embedding_matrix, size=audio_embedding_seq_len, align_corners=True, mode='linear'
        )
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)

    frame_num = vertex_matrix.shape[1]
    audio_embedding_matrix = torch.reshape(
        audio_embedding_matrix, (1, audio_embedding_matrix.shape[1] // factor, audio_embedding_matrix.shape[2] * factor)
    )

    return audio_embedding_matrix, vertex_matrix, frame_num


def predictFrameRateAdjustment(audio_embedding_matrix: torch.Tensor, ifps: int, ofps: int) -> torch.Tensor:
    """Adjusts audio embeddings frame rate for prediction without requiring vertex data.

    This function handles temporal alignment of audio features for prediction scenarios
    where we need to adjust the frame rate but don't have vertex data available.
    It works for all combinations of input and output frame rates.

    Args:
        audio_embedding_matrix (torch.Tensor): Audio feature embeddings from HuBERT model.
            Shape: (batch_size, seq_len, audio_dim)
        ifps (int): Input frame rate (audio sampling rate)
        ofps (int): Output frame rate (target facial animation frame rate)

    Returns:
        torch.Tensor: Adjusted audio embeddings with compatible temporal dimensions.
            Shape: (1, adjusted_seq_len, audio_dim * factor) for compatible rates,
                   (1, interpolated_seq_len, audio_dim) for incompatible rates

    Frame Rate Scenarios:
        1. ifps > ofps (e.g., 50→25): Downsample by factor, concatenate features
        2. ifps < ofps (e.g., 25→50): Upsample by interpolation
        3. ifps = ofps: No change needed
        4. ifps % ofps != 0: Interpolation required

    Example:
        >>> # BIWI: 50fps → 25fps (factor = 2)
        >>> adjusted = predictFrameRateAdjustment(audio_features, ifps=50, ofps=25)
        >>> # VOCASET: 50fps → 60fps (interpolation)
        >>> adjusted = predictFrameRateAdjustment(audio_features, ifps=50, ofps=60)
        >>> # Same rates: 30fps → 30fps (no change)
        >>> adjusted = predictFrameRateAdjustment(audio_features, ifps=30, ofps=30)
    """
    # Handle same frame rates (no adjustment needed)
    if ifps == ofps:
        return audio_embedding_matrix.reshape(1, -1, audio_embedding_matrix.shape[2])

    # Handle compatible frame rates (ifps % ofps == 0)
    if ifps % ofps == 0:
        factor = ifps // ofps

        # Ensure sequence length is divisible by factor
        if audio_embedding_matrix.shape[1] % factor != 0:
            # Truncate to make divisible
            new_length = (audio_embedding_matrix.shape[1] // factor) * factor
            audio_embedding_matrix = audio_embedding_matrix[:, :new_length]

        # Reshape to combine factor frames into one
        new_seq_len = audio_embedding_matrix.shape[1] // factor
        new_feature_dim = audio_embedding_matrix.shape[2] * factor

        return torch.reshape(audio_embedding_matrix, (1, new_seq_len, new_feature_dim))

    # Handle incompatible frame rates (interpolation required)
    else:
        # Calculate target sequence length based on frame rate ratio
        target_seq_len = int(audio_embedding_matrix.shape[1] * (ofps / ifps))

        # Interpolate along the sequence dimension
        # Transpose for interpolation: (batch, seq, features) -> (batch, features, seq)
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)

        # Apply linear interpolation
        audio_embedding_matrix = F.interpolate(
            audio_embedding_matrix, size=target_seq_len, align_corners=True, mode='linear'
        )

        # Transpose back: (batch, features, seq) -> (batch, seq, features)
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)

        # Ensure we return a 3D tensor: (1, seq_len, features)
        # After transpose operations, audio_embedding_matrix is (batch, seq, features)
        return audio_embedding_matrix.reshape(1, -1, audio_embedding_matrix.shape[2])


class FaceXHuBERT(nn.Module):
    """
    FaceXHuBERT model for audio-driven 3D face animation.

    This model takes raw audio input and generates 3D facial vertex sequences, conditioned on subject and emotion embeddings.
    It uses a pretrained HuBERT audio encoder to extract audio features, adjusts the temporal resolution of these features to match
    the target 3D scan frame rate, and then decodes them into 3D vertex positions using a GRU-based sequence model. Subject and emotion
    information are injected via learned linear projections and multiplicative conditioning.

    Attributes:
        dataset (str): Name of the dataset being used.
        i_fps (int): Input frames per second (audio feature rate).
        o_fps (int): Output frames per second (3D scan rate).
        gru_layer_dim (int): Number of GRU layers.
        gru_hidden_dim (int): Hidden dimension size for the GRU.
        audio_encoder (HubertModel): Pretrained HuBERT model for audio feature extraction.
        audio_dim (int): Output feature dimension of the audio encoder.
        gru (nn.GRU): GRU module for temporal modeling of audio features.
        fc (nn.Linear): Fully connected layer mapping GRU output to vertex space.
        obj_vector (nn.Linear): Linear layer for subject (identity) embedding.
        emo_vector (nn.Linear): Linear layer for emotion embedding.
    """

    def __init__(self, args: Any) -> None:
        """
        Initializes the FaceXHuBERT model and its submodules.

        The constructor sets up the audio encoder (HuBERT), freezes selected layers for transfer learning,
        configures the GRU input size based on the relationship between input and output frame rates,
        and initializes the subject and emotion embedding layers.

        Args:
            args (Any): Configuration object with the following required attributes:
                - dataset (str): Dataset name.
                - input_fps (int): Input audio frame rate.
                - output_fps (int): Output 3D scan frame rate.
                - feature_dim (int): Feature dimension for GRU and embeddings.
                - vertice_dim (int): Output dimension for 3D vertices.
                - train_subjects (str): Space-separated list of subject names for training.
        """
        super(FaceXHuBERT, self).__init__()
        self.dataset = args.dataset
        self.i_fps = args.input_fps
        self.o_fps = args.output_fps
        self.gru_layer_dim = 2
        self.gru_hidden_dim = args.feature_dim

        # Load pretrained HuBERT audio encoder and freeze feature extractor
        ckpt = "facebook/hubert-base-ls960"
        self.audio_encoder = HubertModel.from_pretrained(ckpt)
        self.audio_dim = self.audio_encoder.encoder.config.hidden_size
        self.audio_encoder.feature_extractor._freeze_parameters()

        # Freeze early layers of the encoder for stability
        frozen_layers = [0, 1]
        for name, param in self.audio_encoder.named_parameters():
            if name.startswith("feature_projection"):
                param.requires_grad = False
            if name.startswith("encoder.layers"):
                layer = int(name.split(".")[2])
                if layer in frozen_layers:
                    param.requires_grad = False

        # Determine GRU input size: double audio_dim if downsampling by integer factor, else keep as audio_dim
        if args.input_fps % args.output_fps == 0:
            gru_input_size = self.audio_dim * 2
        else:
            gru_input_size = self.audio_dim

        # GRU for temporal modeling of audio features
        self.gru = nn.GRU(gru_input_size, args.feature_dim, self.gru_layer_dim, batch_first=True, dropout=0.3)

        # Output layer: maps GRU output to 3D vertex positions
        self.fc = nn.Linear(args.feature_dim, args.vertice_dim)
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)

        # Subject embedding: projects one-hot subject vector to feature_dim
        num_subjects = len(args.train_subjects.split())
        self.obj_vector = nn.Linear(num_subjects, args.feature_dim, bias=False)

        # Emotion embedding: projects one-hot emotion vector to feature_dim
        self.emo_vector = nn.Linear(2, args.feature_dim, bias=False)

    def forward(
        self,
        audio: Tensor,
        template: Tensor,
        vertice: Tensor,
        one_hot: Tensor,
        emo_one_hot: Tensor,
        criterion: Callable[[Tensor, Tensor], Tensor],
    ) -> Tensor:
        """
        Forward pass for training.

        This method processes the input audio and template, applies subject and emotion conditioning,
        adjusts the temporal resolution of the audio features to match the target frame rate,
        and predicts the 3D vertex sequence. The loss is computed against the ground truth vertices.

        Args:
            audio (Tensor): Input audio waveform tensor of shape (batch_size, raw_wav).
            template (Tensor): Template mesh tensor of shape (batch_size, V*3).
            vertice (Tensor): Ground truth vertex tensor of shape (batch_size, seq_len, V*3).
            one_hot (Tensor): One-hot subject embedding tensor of shape (batch_size, num_subjects).
            emo_one_hot (Tensor): One-hot emotion embedding tensor of shape (batch_size, 2).
            criterion (Callable): Loss function to compare prediction and ground truth.

        Returns:
            Tensor: Scalar loss value (averaged over batch and sequence).
        """
        # Expand template to match sequence length for residual connection
        template = template.unsqueeze(1)
        # Project subject and emotion one-hot vectors to feature space
        obj_embedding = self.obj_vector(one_hot)
        emo_embedding = self.emo_vector(emo_one_hot)
        # Extract audio features using HuBERT
        hidden_states = self.audio_encoder(audio).last_hidden_state

        # Adjust audio features and ground truth vertices to match output frame rate
        hidden_states, vertice, frame_num = inputRepresentationAdjustment(
            hidden_states, vertice, self.i_fps, self.o_fps
        )
        # Truncate to the minimum valid sequence length
        hidden_states = hidden_states[:, :frame_num]

        # Initialize GRU hidden state
        h0 = torch.zeros(self.gru_layer_dim, hidden_states.shape[0], self.gru_hidden_dim).requires_grad_().cuda()

        # Sequence modeling with GRU
        vertice_out, _ = self.gru(hidden_states, h0)
        # Apply subject and emotion conditioning (elementwise multiplication)
        vertice_out = vertice_out * obj_embedding
        vertice_out = vertice_out * emo_embedding

        # Map to 3D vertex space
        vertice_out = self.fc(vertice_out)
        # Add template mesh as a residual
        vertice_out = vertice_out + template

        # Compute and return loss
        loss = criterion(vertice_out, vertice)
        loss = torch.mean(loss)
        return loss

    def predict(self, audio: Tensor, template: Tensor, one_hot: Tensor, emo_one_hot: Tensor) -> Tensor:
        """
        Inference/prediction pass for generating 3D face animation from audio.

        This method processes the input audio and template, applies subject and emotion conditioning,
        adjusts the temporal resolution of the audio features to match the target frame rate,
        and predicts the 3D vertex sequence. No loss is computed.

        Args:
            audio (Tensor): Input audio waveform tensor of shape (batch_size, raw_wav).
            template (Tensor): Template mesh tensor of shape (batch_size, V*3).
            one_hot (Tensor): One-hot subject embedding tensor of shape (batch_size, num_subjects).
            emo_one_hot (Tensor): One-hot emotion embedding tensor of shape (batch_size, 2).

        Returns:
            Tensor: Predicted vertex tensor of shape (batch_size, seq_len, V*3).
        """
        # Expand template to match sequence length for residual connection
        template = template.unsqueeze(1)
        # Project subject and emotion one-hot vectors to feature space
        obj_embedding = self.obj_vector(one_hot)
        emo_embedding = self.emo_vector(emo_one_hot)
        # Extract audio features using HuBERT
        hidden_states = self.audio_encoder(audio).last_hidden_state

        # Adjust audio features to match output frame rate
        hidden_states = predictFrameRateAdjustment(hidden_states, self.i_fps, self.o_fps)

        # Initialize GRU hidden state
        h0 = torch.zeros(self.gru_layer_dim, hidden_states.shape[0], self.gru_hidden_dim).requires_grad_().cuda()

        # Sequence modeling with GRU
        vertice_out, _ = self.gru(hidden_states, h0)
        # Apply subject and emotion conditioning (elementwise multiplication)
        vertice_out = vertice_out * obj_embedding
        vertice_out = vertice_out * emo_embedding

        # Map to 3D vertex space
        vertice_out = self.fc(vertice_out)
        # Add template mesh as a residual
        vertice_out = vertice_out + template

        return vertice_out
