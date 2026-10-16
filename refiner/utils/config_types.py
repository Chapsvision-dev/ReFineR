from pathlib import Path
from enum import IntEnum
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from omegaconf import MISSING

from refiner.camera.backends import IntrinsicsBackend

class RenderType(IntEnum):
    GS = 0

@dataclass
class OptimCfg:
    lr: float = MISSING
    niter: Tuple[float, int, int] = MISSING


@dataclass
class RerunSaverCfg:
    path: str = ""
    pc_channel: str = ""
    spawn: bool = True
    modules: Tuple[str, ...] = ()
    view_coordinates: str = "RIGHT_HAND_Z_UP"


@dataclass
class FileSaverCfg:
    path: str = MISSING
    pc_channel: str = MISSING
    modules: Tuple[str, ...] = ()


@dataclass
class HOGLayerCfg:
    nbins: int = MISSING
    cell_size: int = MISSING
    cells_per_block: int = MISSING


@dataclass
class CompCfg:
    comp: Tuple[str, str] = MISSING
    weight: int = MISSING


@dataclass
class HOGLossCfg:
    channels: List[CompCfg] = MISSING
    weight: float = MISSING
    layers: List[HOGLayerCfg] = MISSING


@dataclass
class BasicLossCfg:
    channels: List[CompCfg] = MISSING
    weight: float = MISSING


@dataclass
class LossCfg:
    hogs: Optional[HOGLossCfg] = None
    photo: Optional[BasicLossCfg] = None
    grad: Optional[BasicLossCfg] = None
    ma: Optional[BasicLossCfg] = None


@dataclass
class GSCfg:
    point_size: float = MISSING
    opacity: float = MISSING
    mip_size: float = 2


@dataclass
class PertCfg:
    p: Optional[float] = None
    r: Optional[float] = None
    t: Optional[float] = None
    f: Optional[float] = None
    c: Optional[float] = None


@dataclass
class DataCfg:
    set: str = MISSING
    scene: str = MISSING

    path: str = MISSING
    precomp_path: str = ""

    pc_channels: List[str] = (MISSING,)
    pc_subsample_factor: float = 1.0
    pc_dynamic_scale: bool = True
    pc_view_filter: Tuple[float, float, float] = (20, 1, 100)
    pc_lod_filter: float = -1
    pc_normalize_intensity: bool = False

    img_channels: List[str] = MISSING
    img_min_size: int = -1
    img_center_crop: Tuple[int, int] = (0, 0)
    equalize_intensity: bool = True
    absolute_normals: bool = True

    pert_init: Optional[PertCfg] = None
    seed: int = -1


@dataclass
class CameraCfg:
    extr_grad: bool = MISSING
    intr_grad: bool = MISSING
    dist_grad: bool = False
    
    clip_plane: Tuple[float, float] = MISSING
    ndc_ranges: Tuple[int, int, int, int, int, int] = (-1, 1, 1, -1, 0, 1)
    viewport: Tuple[int, int, int, int, int, int] = (0, -1, -1, 0, 0, 1)
    intr_model: IntrinsicsBackend.MODEL = IntrinsicsBackend.MODEL.FXFY


@dataclass
class LogCfg:
    log_freq: int
    log_to_file: bool = False
    log_to_console: bool = True

@dataclass
class RenderCfg:
    render_type: RenderType = RenderType.GS
    render_channels: List[str] = MISSING

@dataclass
class RefineCfg:
    saver_rerun: Optional[RerunSaverCfg] = None
    saver_file: Optional[FileSaverCfg] = None

    camera: CameraCfg = MISSING
    gs: GSCfg = MISSING
    renderer: RenderCfg = MISSING
    loss: LossCfg = MISSING
    optim: OptimCfg = MISSING

    out_path: Path = Path("output/main")
    matchAnything_init: bool = False
    matchAnything_init_gs: bool = True


@dataclass
class FullCfg(RefineCfg):
    data: DataCfg = MISSING
    logger: LogCfg = MISSING


@dataclass
class MainCfg:
    include: List[str] = field(default_factory=lambda: [])
