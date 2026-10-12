from typing import List, Final

from dataclasses import dataclass, field

# In python 3.11 this could be replaced with StrEnum
@dataclass(frozen=True)
class Channels:
    normals_base_x = "nx0"
    normals_base_y = "ny0"
    normals_base_z = "nz0"
    normals_base_mean = "n0"

    normals_x = "nx"
    normals_y = "ny"
    normals_z = "nz"
    normals_mean = "n"

    normals_pos_x = "npx"
    normals_pos_y = "npy"
    normals_pos_z = "npz"
    normals_pos_mean = "np"

    reflectivity = "r"
    reflectivity_base = "r0"

    red = "red"
    green = "green"
    blue = "blue"
    gray = "g"

    red_base = "red0"
    green_base = "green0"
    blue_base = "blue0"
    gray_base = "g0"

    scale_1 = "sx"
    scale_2 = "sy"
    scale_3 = "sz"

    rot_x = "rx"
    rot_y = "ry"
    rot_z = "rz"

    linearity = "lin"
    planarity = "plan"
    sphericity = "sph"

    x = "x"
    y = "y"
    z = "z"

    xp = "xp"
    yp = "yp"

    xi = "xi"
    yi = "yi"

    albedo_red = "ar"
    albedo_green = "ag"
    albedo_blue = "ab"
    albedo_mean = "alb"

    material_red = "mr"
    material_green = "mg"
    material_blue = "mb"
    material_mean = "mat"

    depth = "d"
    depth_inv = "di"
    depth_norm = "dn"
    depth_inv_norm = "dni"
    opacity = "o"

    seg_red = "seg_red"
    seg_green = "seg_gree"
    seg_blue = "seg_blue"
    seg = "seg"
    seg_base = "seg0"

    idx = "idx"

CHANNELS = Channels()

@dataclass(frozen=True)
class ChannelsGroups:
    normals_base: Final[List[str]] = field(default_factory=lambda: [
        CHANNELS.normals_base_x,
        CHANNELS.normals_base_y,
        CHANNELS.normals_base_z])
    
    normals: Final[List[str]] = field(default_factory=lambda: [
        CHANNELS.normals_x,
        CHANNELS.normals_y,
        CHANNELS.normals_z])

    normals_pos: Final[List[str]] = field(default_factory=lambda: [
        CHANNELS.normals_pos_x,
        CHANNELS.normals_pos_y,
        CHANNELS.normals_pos_z])
    
    colors: Final[List[str]] = field(default_factory=lambda: [
        CHANNELS.red, 
        CHANNELS.green, 
        CHANNELS.blue])
    
    colors_base: Final[List[str]] = field(default_factory=lambda: [
        CHANNELS.red_base,
        CHANNELS.green_base,
        CHANNELS.blue_base])
    
    rot: Final[List[str]] = field(
        default_factory=lambda: [
            CHANNELS.rot_x, 
            CHANNELS.rot_y, 
            CHANNELS.rot_z])
    
    xyz: Final[List[str]] = field(default_factory=lambda: [
        CHANNELS.x,
        CHANNELS.y,
        CHANNELS.z])

    img_xy: Final[List[str]] = field(default_factory=lambda: [
        CHANNELS.xp,
        CHANNELS.yp])

    img_coords: Final[List[str]] = field(default_factory=lambda: [
    CHANNELS.xi,
    CHANNELS.yi])

    
    albedo: Final[List[str]] = field(default_factory=lambda: [
        CHANNELS.albedo_red, 
        CHANNELS.albedo_green,
        CHANNELS.albedo_blue
    ])

    material: Final[List[str]] = field(default_factory=lambda: [
        CHANNELS.material_red,
        CHANNELS.material_green,
        CHANNELS.material_blue])
    
    depth: Final[List[str]] = field(default_factory=lambda: [
        CHANNELS.depth, 
        CHANNELS.depth_inv, 
        CHANNELS.depth_inv_norm, 
        CHANNELS.depth_norm])
    
    seg_city: Final[List[str]] = field(default_factory=lambda: [
        CHANNELS.seg_red, 
        CHANNELS.seg_green, 
        CHANNELS.seg_blue]
    )

    synth: Final[List[str]] = field(default_factory=lambda: [
        CHANNELS.depth, 
        CHANNELS.opacity,
        CHANNELS.xi,
        CHANNELS.yi,
        CHANNELS.idx
    ])

CHANNELS_GROUPS = ChannelsGroups()