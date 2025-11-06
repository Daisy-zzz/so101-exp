from experiments.robot.openvla_utils import (
    get_vla,
    get_processor,
    get_action_head,
    get_proprio_projector,
    get_vla_action,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

from lerobot.policies.adapter.configuration_adapter import AdapterConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from collections import deque
import torch
from torch import Tensor

def encode_obs(obs: dict) -> dict:
    return {
        "full_image": obs["observation"]["images.top"],
        "wrist_image": obs["observation"]["images.wrist"],
        "state": obs["observation"]["state"],
        "instruction": obs["task"],
    }


class Model:
    def __init__(self, cfg: AdapterConfig):
        self.cfg = cfg
        self.vla = get_vla(cfg)
        self.processor = get_processor(cfg)
        self.action_head = None
        if cfg.use_l1_regression:
            self.action_head = get_action_head(cfg, self.vla.llm_dim)
        self.proprio_projector = None
        if cfg.use_proprio:
            self.proprio_projector = get_proprio_projector(
                cfg, self.vla.llm_dim, PROPRIO_DIM
            )

    def get_action(self, observation: dict):
        obs = encode_obs(observation)
        actions = get_vla_action(
            cfg=self.cfg,
            vla=self.vla,
            processor=self.processor,
            obs=obs,
            task_label=obs["instruction"],
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            use_film=self.cfg.use_film,
        )
        return actions
    
class AdapterPolicy(PreTrainedPolicy):
    """Wrapper around external Adapter model for LeRobot integration."""

    name = "adapter"
    config_class = AdapterConfig

    def __init__(self, config: AdapterConfig):
        """Initialize Adapter policy wrapper."""
        super().__init__(config)

        self.cfg = config
        self.vla = get_vla(config)
        self.processor = get_processor(config, self.vla.tokenizer)
        self.action_head = None
        if config.use_l1_regression:
            self.action_head = get_action_head(config, self.vla.llm_dim)
        self.proprio_projector = None
        if config.use_proprio:
            self.proprio_projector = get_proprio_projector(
                config, self.vla.llm_dim, PROPRIO_DIM
            )
    
    def _create_adapter_model(self):
        config_args = {
            "pretrained_checkpoint": self.cfg.pretrained_checkpoint,
            "use_l1_regression": self.cfg.use_l1_regression,
            "use_diffusion": self.cfg.use_diffusion,
            "use_film": self.cfg.use_film,
            "use_proprio": self.cfg.use_proprio,
            "load_in_8bit": self.cfg.load_in_8bit,
            "load_in_4bit": self.cfg.load_in_4bit,
            "num_images_in_input": self.cfg.num_images_in_input,
            "center_crop": self.cfg.center_crop,
            "unnorm_key": self.cfg.unnorm_key,
            "chunk_size": self.cfg.chunk_size,
            "n_action_steps": self.cfg.n_action_steps,
            "lora_rank": self.cfg.lora_rank,
        }

        cfg = AdapterConfig(**config_args)
        return Model(cfg)

    def reset(self):
        """Reset policy state when environment resets."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions from the model given a batch of observations."""
        obs = encode_obs(batch)
        actions = get_vla_action(
            cfg=self.cfg,
            vla=self.vla,
            processor=self.processor,
            obs=obs,
            task_label=obs["instruction"],
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            use_film=self.cfg.use_film,
        )
        return actions
    
    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select single action from action queue."""
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
    

    


    