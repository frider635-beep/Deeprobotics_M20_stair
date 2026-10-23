"""Stable terrain-column assignment for Stage9 width generalization."""
from copy import deepcopy

ASCENT_TASKS = (
    (.20, 1.0), (.20, 1.2),
    (.25, 1.0), (.25, 1.2),
    (.30, 1.0), (.30, 1.2),
    (.35, 1.0), (.35, 1.2),
)
REFERENCE_TASK = (.30, 5.0)
DESCENT_TASKS = ASCENT_TASKS
GENERAL_TERRAINS = ('boxes', 'random_rough', 'hf_pyramid_slope', 'hf_pyramid_slope_inv')
FLAT_COPIES = 3
GENERAL_COPIES = 2

ASCENT_COLUMNS = len(ASCENT_TASKS) + 1
ASCENT_REFERENCE_COLUMN = len(ASCENT_TASKS)
FLAT_START = ASCENT_COLUMNS
FLAT_COLUMNS = tuple(range(FLAT_START, FLAT_START+FLAT_COPIES))
FLAT_COLUMN = FLAT_COLUMNS[0]
DESCENT_START = FLAT_START + FLAT_COPIES
DESCENT_COLUMNS = tuple(range(DESCENT_START, DESCENT_START+len(DESCENT_TASKS)))
GENERAL_START = DESCENT_START + len(DESCENT_TASKS)
GENERAL_COLUMN_GROUPS = {
    name: tuple(range(GENERAL_START+i*GENERAL_COPIES, GENERAL_START+(i+1)*GENERAL_COPIES))
    for i, name in enumerate(GENERAL_TERRAINS)
}
GENERAL_COLUMNS = tuple(column for columns in GENERAL_COLUMN_GROUPS.values() for column in columns)
TERRAIN_COLUMNS = GENERAL_START + len(GENERAL_TERRAINS)*GENERAL_COPIES
TASK_WIDTHS_M = tuple(sorted({width for _, width in ASCENT_TASKS}))

BASELINE_NUM_ENVS = 8192
STAGED_NUM_ENVS = 10584
NON_STAIR_FRACTION = (len(FLAT_COLUMNS) + len(GENERAL_COLUMNS)) / 28 * BASELINE_NUM_ENVS / STAGED_NUM_ENVS
DESCENT_FRACTION = len(DESCENT_COLUMNS) / 28 * BASELINE_NUM_ENVS / STAGED_NUM_ENVS
ASCENT_FRACTION = 1. - NON_STAIR_FRACTION - DESCENT_FRACTION

ASCENT_EASY_COLUMNS = (5, 7, ASCENT_REFERENCE_COLUMN)
ASCENT_MIDDLE_COLUMNS = (3, 4, 6)
ASCENT_HARD_COLUMNS = (0, 1, 2)


def staged_sampling_probabilities(stage):
    """Per-column episode sampling, preserving baseline non-ascent counts."""
    probabilities = [0.] * TERRAIN_COLUMNS
    if stage == 0:
        groups, group_shares = (ASCENT_EASY_COLUMNS, ASCENT_MIDDLE_COLUMNS, ASCENT_HARD_COLUMNS), (.70, .20, .10)
    elif stage == 1:
        groups, group_shares = (ASCENT_EASY_COLUMNS, ASCENT_MIDDLE_COLUMNS, ASCENT_HARD_COLUMNS), (.45, .35, .20)
    elif stage == 2:
        groups, group_shares = (tuple(range(ASCENT_COLUMNS)),), (1.,)
    else:
        raise ValueError(f"Unknown Stage9 sampling stage: {stage}")
    for columns, share in zip(groups, group_shares):
        for column in columns:
            probabilities[column] = ASCENT_FRACTION * share / len(columns)
    for column in DESCENT_COLUMNS:
        probabilities[column] = DESCENT_FRACTION / len(DESCENT_COLUMNS)
    for column in (*FLAT_COLUMNS, *GENERAL_COLUMNS):
        probabilities[column] = NON_STAIR_FRACTION / (len(FLAT_COLUMNS) + len(GENERAL_COLUMNS))
    return tuple(probabilities)

if (ASCENT_REFERENCE_COLUMN != 8 or FLAT_COLUMNS != (9, 10, 11)
        or DESCENT_COLUMNS != tuple(range(12, 20))
        or GENERAL_COLUMNS != tuple(range(20, 28)) or TERRAIN_COLUMNS != 28):
    raise RuntimeError("Stage9 terrain column layout changed unexpectedly")
for _stage in range(3):
    if abs(sum(staged_sampling_probabilities(_stage)) - 1.) > 1.e-9:
        raise RuntimeError("Stage9 staged terrain probabilities must sum to one")


def task_label(direction, depth, width):
    return f"{direction}_d{int(round(depth * 100)):03d}_w{int(round(width * 100)):03d}"


def preserve_general_terrains(source_terrains):
    missing = [name for name in GENERAL_TERRAINS if name not in source_terrains]
    if missing:
        raise RuntimeError("Stage9 mixed-terrain source is missing official terrains: " + ", ".join(missing))
    return {name: deepcopy(source_terrains[name]) for name in GENERAL_TERRAINS}
