import torch

from third_party.MatchAnything.third_party.ROMA.roma.matchanything_roma_model import MatchAnything_Model
from third_party.MatchAnything.src.config.default import get_cfg_defaults
from third_party.MatchAnything.src.utils.misc import lower_config
from third_party.MatchAnything.third_party.ROMA.roma.models.encoders import CNNandDinov2

def get_ma_encoder() -> CNNandDinov2:
    main_cfg_path = "third_party/MatchAnything/configs/models/roma_model.py"
    chkpt = "weights/MatchAnything/matchanything_roma.ckpt"

    config =  get_cfg_defaults()
    config.merge_from_file(main_cfg_path)    
    config = lower_config(config)

    model = MatchAnything_Model(config=config['roma'], test_mode=True)
    state_dict = torch.load(chkpt, map_location='cpu')['state_dict']
    model.load_state_dict(state_dict, strict=False)
    model = model.to(dtype = torch.float32, device = "cuda:0")

    return model.model.encoder

