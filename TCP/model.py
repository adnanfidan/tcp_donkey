from collections import deque
import numpy as np
import torch 
from torch import nn
from TCP.resnet import *


class PIDController(object):
	def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
		self._K_P = K_P
		self._K_I = K_I
		self._K_D = K_D

		self._window = deque([0 for _ in range(n)], maxlen=n)
		self._max = 0.0
		self._min = 0.0

	def step(self, error):
		self._window.append(error)
		self._max = max(self._max, abs(error))
		self._min = -abs(self._max)

		if len(self._window) >= 2:
			integral = np.mean(self._window)
			derivative = (self._window[-1] - self._window[-2])
		else:
			integral = 0.0
			derivative = 0.0

		return self._K_P * error + self._K_I * integral + self._K_D * derivative
      

class TCP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
        self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)

        self.perception = resnet34(pretrained=True)

        self.measurements = nn.Sequential(
            nn.Linear(1+2+6, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )

        self.join_traj = nn.Sequential(
            nn.Linear(128+1000, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )

        self.join_ctrl = nn.Sequential(
            nn.Linear(128+512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )

        self.speed_branch = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.value_branch_traj = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.value_branch_ctrl = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        # ---- eski mu/sigma başları yerine aksiyon başı
        self.policy_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True),
        )
        # Aksiyon üretimi (2: acc_like, steer)
        self.action_head = nn.Linear(256, 2)

        # ---- KONTROL GRU: input_size 256'ya indi (mu/sigma yok)
        self.decoder_ctrl = nn.GRUCell(input_size=256, hidden_size=256)

        self.output_ctrl = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )

        # self.dist_mu / self.dist_sigma KALDIRILDI

        self.decoder_traj = nn.GRUCell(input_size=4, hidden_size=256)
        self.output_traj = nn.Linear(256, 2)

        self.init_att = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10*8),
            nn.Softmax(1)
        )
        self.wp_att = nn.Sequential(
            nn.Linear(256+256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10*8),
            nn.Softmax(1)
        )
        self.merge = nn.Sequential(
            nn.Linear(512+256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
        )

    def forward(self, img, state, target_point):
        feature_emb, cnn_feature = self.perception(img)
        outputs = {}
        outputs['pred_speed'] = self.speed_branch(feature_emb)
        measurement_feature = self.measurements(state)

        # ----- Traj
        j_traj = self.join_traj(torch.cat([feature_emb, measurement_feature], 1))
        outputs['pred_value_traj'] = self.value_branch_traj(j_traj)
        outputs['pred_features_traj'] = j_traj
        z = j_traj
        output_wp, traj_hidden_state = [], []

        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).type_as(z)
        for _ in range(self.config.pred_len):
            x_in = torch.cat([x, target_point], dim=1)
            z = self.decoder_traj(x_in, z)
            traj_hidden_state.append(z)
            dx = self.output_traj(z)
            x = dx + x
            output_wp.append(x)

        pred_wp = torch.stack(output_wp, dim=1)
        outputs['pred_wp'] = pred_wp

        traj_hidden_state = torch.stack(traj_hidden_state, dim=1)
        init_att = self.init_att(measurement_feature).view(-1, 1, 8, 10)
        feature_emb = torch.sum(cnn_feature * init_att, dim=(2, 3))
        j_ctrl = self.join_ctrl(torch.cat([feature_emb, measurement_feature], 1))
        outputs['pred_value_ctrl'] = self.value_branch_ctrl(j_ctrl)
        outputs['pred_features_ctrl'] = j_ctrl

        # ----- Kontrol döngüsü (mu/sigma yok)
        x = j_ctrl
        h = torch.zeros(size=(x.shape[0], 256), dtype=x.dtype).type_as(x)

        actions_seq = []
        future_features = []

        for t in range(self.config.pred_len):
            # artık sadece x'i besliyoruz
            h = self.decoder_ctrl(x, h)

            wp_att = self.wp_att(torch.cat([h, traj_hidden_state[:, t]], 1)).view(-1, 1, 8, 10)
            new_feature_emb = torch.sum(cnn_feature * wp_att, dim=(2, 3))

            merged_feature = self.merge(torch.cat([h, new_feature_emb], 1))
            dx = self.output_ctrl(merged_feature)
            x = dx + x

            # aksiyon üretimi (deterministik)
            policy_feat = self.policy_head(x)
            action_logits = self.action_head(policy_feat)          # [B,2]
            action = torch.tanh(action_logits)                     # [-1,1] iki kanal
            # channel 0: acc_like (-1..1), channel 1: steer (-1..1)

            actions_seq.append(action)
            future_features.append(x)

        outputs['actions_seq'] = actions_seq            # liste, [pred_len x (B,2)]
        outputs['action'] = actions_seq[-1]             # son adım aksiyonu (B,2)
        outputs['future_feature'] = future_features

        return outputs

    def process_action(self, pred, command, speed, target_point):
        """
        Beklenen: pred['action'] ∈ [-1,1]^2, sırası [acc_like, steer]
        """
        acc_like, steer = pred['action'][:, 0], pred['action'][:, 1]

        # acc_like >= 0 -> throttle, <0 -> brake
        throttle = torch.clamp(acc_like, min=0.0, max=1.0)
        brake = torch.clamp(-acc_like, min=0.0, max=1.0)   # negatifse fren miktarı
        steer = torch.clamp(steer, -1.0, 1.0)

        # tensörleri numpy/float'a çevir
        throttle_v = float(throttle[0].detach().cpu().numpy().astype(np.float64))
        steer_v    = float(steer[0].detach().cpu().numpy().astype(np.float64))
        brake_v    = float(brake[0].detach().cpu().numpy().astype(np.float64))

        metadata = {
            'speed': float(speed.cpu().numpy().astype(np.float64)),
            'steer': steer_v,
            'throttle': throttle_v,
            'brake': brake_v,
            'command': command,
            'target_point': tuple(target_point[0].data.cpu().numpy().astype(np.float64)),
        }
        return steer_v, throttle_v, brake_v, metadata
