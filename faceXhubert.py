import torch
import torch.nn as nn
import torch.nn.functional as F

from hubert.modeling_hubert import HubertModel


def inputRepresentationAdjustment(audio_embedding_matrix, vertex_matrix, ifps, ofps):
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


class FaceXHuBERT(nn.Module):
    def __init__(self, args):
        super(FaceXHuBERT, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        self.dataset = args.dataset
        self.i_fps = args.input_fps  # audio fps (input to the network)
        self.o_fps = args.output_fps  # 4D Scan fps (output or target)
        self.gru_layer_dim = 2
        self.gru_hidden_dim = args.feature_dim

        # Audio Encoder
        ckpt = "facebook/hubert-base-ls960"
        # ckpt = "facebook/hubert-large-ls960-ft"
        # ckpt = "facebook/hubert-xlarge-ls960-ft"
        self.audio_encoder = HubertModel.from_pretrained(ckpt)
        self.audio_dim = self.audio_encoder.encoder.config.hidden_size
        self.audio_encoder.feature_extractor._freeze_parameters()

        frozen_layers = [0, 1]

        for name, param in self.audio_encoder.named_parameters():
            if name.startswith("feature_projection"):
                param.requires_grad = False
            if name.startswith("encoder.layers"):
                layer = int(name.split(".")[2])
                if layer in frozen_layers:
                    param.requires_grad = False

        # Vertex Decoder
        # GRU module - Dynamic input size based on FPS ratio
        if args.input_fps % args.output_fps == 0:
            # Factor-based adjustment (like BIWI: 50->25) - audio features are concatenated
            gru_input_size = self.audio_dim * 2
        else:
            # Interpolation-based adjustment (like VOCASET: 50->60) - audio features stay same size
            gru_input_size = self.audio_dim

        self.gru = nn.GRU(gru_input_size, args.feature_dim, self.gru_layer_dim, batch_first=True, dropout=0.3)
        # Fully connected layer
        self.fc = nn.Linear(args.feature_dim, args.vertice_dim)
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)

        # Subject embedding, S - Dynamic based on number of training subjects
        num_subjects = len(args.train_subjects.split())
        self.obj_vector = nn.Linear(num_subjects, args.feature_dim, bias=False)

        # Emotion embedding, E
        self.emo_vector = nn.Linear(2, args.feature_dim, bias=False)

    def forward(self, audio, template, vertice, one_hot, emo_one_hot, criterion):

        template = template.unsqueeze(1)
        obj_embedding = self.obj_vector(one_hot)
        emo_embedding = self.emo_vector(emo_one_hot)
        hidden_states = audio
        hidden_states = self.audio_encoder(hidden_states).last_hidden_state

        hidden_states, vertice, frame_num = inputRepresentationAdjustment(
            hidden_states, vertice, self.i_fps, self.o_fps
        )

        hidden_states = hidden_states[:, :frame_num]

        h0 = torch.zeros(self.gru_layer_dim, hidden_states.shape[0], self.gru_hidden_dim).requires_grad_().cuda()

        vertice_out, _ = self.gru(hidden_states, h0)
        vertice_out = vertice_out * obj_embedding
        vertice_out = vertice_out * emo_embedding

        vertice_out = self.fc(vertice_out)

        vertice_out = vertice_out + template

        loss = criterion(vertice_out, vertice)
        loss = torch.mean(loss)
        return loss

    def predict(self, audio, template, one_hot, emo_one_hot):
        template = template.unsqueeze(1)
        obj_embedding = self.obj_vector(one_hot)
        emo_embedding = self.emo_vector(emo_one_hot)
        hidden_states = audio
        hidden_states = self.audio_encoder(hidden_states).last_hidden_state

        # Apply the same audio processing logic as in forward() based on FPS ratio
        if self.i_fps % self.o_fps == 0:
            # Factor-based adjustment (like BIWI: 50->25) - concatenate features
            if hidden_states.shape[1] % 2 != 0:
                hidden_states = hidden_states[:, : hidden_states.shape[1] - 1]
            hidden_states = torch.reshape(hidden_states, (1, hidden_states.shape[1] // 2, hidden_states.shape[2] * 2))
        # For interpolation-based adjustment (like VOCASET: 50->60), keep features as-is

        h0 = torch.zeros(self.gru_layer_dim, hidden_states.shape[0], self.gru_hidden_dim).requires_grad_().cuda()

        vertice_out, _ = self.gru(hidden_states, h0)
        vertice_out = vertice_out * obj_embedding
        vertice_out = vertice_out * emo_embedding

        vertice_out = self.fc(vertice_out)

        vertice_out = vertice_out + template

        return vertice_out
