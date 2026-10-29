"""
四地車牌合成數據生成器
"""
from .plate_specs import (
    PlateSpec,
    generate_plate,
    generate_mainland_plate,
    generate_gangao_plate,
    generate_hk_plate,
    generate_mo_plate,
    PROVINCES,
    MAINLAND_ALPHA,
)
from .plate_renderer import render_plate, render_random_plate
from .augmentations import (
    augment_plate,
    simulate_capture_condition,
    add_occlusion,
)
