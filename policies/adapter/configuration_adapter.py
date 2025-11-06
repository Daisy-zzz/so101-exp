from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM
from prismatic.models.action_heads import L1RegressionActionHead
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)
from lerobot.utils.constants import OBS_IMAGES

@PreTrainedConfig.register_subclass("adapter")
@dataclass
class AdapterConfig(PreTrainedConfig):
    pretrained_checkpoint: str = "checkpoints/configs+pickcube_so101+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--pickcube_so101----20000_chkpt"
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    use_proprio: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    lora_rank: int = 64
    num_images_in_input: int = 2
    center_crop: bool = True
    unnorm_key: str = "pickcube_so101"
    chunk_size: int = NUM_ACTIONS_CHUNK
    n_action_steps: int = NUM_ACTIONS_CHUNK
    
    def __post_init__(self):
        super().__post_init__()

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None