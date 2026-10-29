"""Fixed minority of narrow ascent practice, entirely outside all curricula."""
import torch
from . import stair_path_layout as layout
from .stair_path_course import PREFIXES
from ..ascent.ascent_adaptive import HEIGHTS

SETTINGS = dict(heights_m=list(HEIGHTS), sampling='uniform_random_at_episode_reset',
    prefixes='fixed_full_flight18', own_outcomes='rewards_and_diagnostics_only',
    height_curriculum=False,prefix_curriculum=False,sampling_readiness=False,
    linked_to_other_course=False,descent='unchanged')


def resample_height_practice(terrain, course, ids, eligible):
    """Fixed full task and random height; neither own nor wider outcomes enter."""
    narrow = torch.isin(terrain.terrain_types[ids],ids.new_tensor(layout.NARROW_ASCENT_COLUMNS))
    practice = ids[narrow]
    course.skill[practice] = PREFIXES.index(18)
    # Old source-run windows cannot later act as curriculum evidence.
    course._clear_evidence(practice)
    course.last_events[practice] = 0;course.evidence_last[practice] = 0
    selected = ids[narrow & eligible]
    if not len(selected):
        return selected
    new_rows = torch.randint(len(HEIGHTS),(len(selected),),device=ids.device)
    changed = selected[new_rows != terrain.terrain_levels[selected]]
    terrain.terrain_levels[selected] = new_rows
    terrain.env_origins[selected] = terrain.terrain_origins[new_rows,terrain.terrain_types[selected]]
    return changed
