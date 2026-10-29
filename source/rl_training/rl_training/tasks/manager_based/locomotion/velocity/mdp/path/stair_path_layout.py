"""Stable terrain-column assignment for Stage9 width generalization."""
from copy import deepcopy
from bisect import bisect_right
from math import floor

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
NARROW_ASCENT_COLUMNS = tuple(c for c,(_,w) in enumerate(ASCENT_TASKS) if w == 1.0)
NARROW_ASCENT_PRACTICE_SHARE = .05
NARROW_ASCENT_SETTINGS = dict(width_m=1.0, columns=list(NARROW_ASCENT_COLUMNS),
    fraction_of_ascent_pool=NARROW_ASCENT_PRACTICE_SHARE, each_depth_fraction=.0125,
    transfer='surplus_to_same_depth_1p2m_column', terrain_height='uniform_random_all_six_rows_at_reset',
    prefix_course='disabled_fixed_full_flight18', sampling_readiness='excluded',
    height_course='disabled_no_link_to_any_mastery_or_outcomes',
    reward_and_physical_failure='unchanged', descent='unchanged')

BASELINE_NUM_ENVS = 8192
STAGED_NUM_ENVS = 10584
NON_STAIR_FRACTION = (len(FLAT_COLUMNS) + len(GENERAL_COLUMNS)) / 28 * BASELINE_NUM_ENVS / STAGED_NUM_ENVS
DESCENT_FRACTION = len(DESCENT_COLUMNS) / 28 * BASELINE_NUM_ENVS / STAGED_NUM_ENVS
ASCENT_FRACTION = 1. - NON_STAIR_FRACTION - DESCENT_FRACTION

ASCENT_EASY_COLUMNS = (5, 7, ASCENT_REFERENCE_COLUMN)
ASCENT_MIDDLE_COLUMNS = (3, 4, 6)
ASCENT_HARD_COLUMNS = (0, 1, 2)
LEGACY_SAMPLING_VERSION = 'm20_stair_path_fixed247_20261023_v3_sampling_40k'
SAMPLING_VERSION = 'm20_stair_path_fixed247_20261026_v4_readiness_sampling'
SAMPLING_TOTAL_ITERATIONS = 40000
SAMPLING_BOUNDARIES = (3000, 6000, 10000, 14000, 20000)
SAMPLING_TRANSITION_ITERATIONS = 500
SAMPLING_GROUP_SHARES = (
    (.70, .20, .10), (.60, .25, .15), (.50, .30, .20),
    (.40, .35, .25), (.35, .35, .30), (1./3, 1./3, 1./3),
)
# Intermediate quotas limit each transfer to at most 5% of the ascent pool.
READINESS_GROUP_SHARES = tuple(
    tuple((a+b)/2 for a,b in zip(left,right)) if half else left
    for left,right in zip(SAMPLING_GROUP_SHARES, SAMPLING_GROUP_SHARES[1:])
    for half in (False, True)
) + (SAMPLING_GROUP_SHARES[-1],)


def sampling_schedule(iteration):
    """Target phase and linear blend; geometry/height promotion stays separate."""
    iteration = max(int(iteration), 0)
    stage = bisect_right(SAMPLING_BOUNDARIES, iteration)
    blend = (min((iteration-SAMPLING_BOUNDARIES[stage-1])/SAMPLING_TRANSITION_ITERATIONS, 1.)
             if stage else 1.)
    return stage, blend


def integer_sample_counts(probabilities, count):
    """Largest-remainder allocation gives exact, reproducible population quotas."""
    scaled = [float(p)*count for p in probabilities]
    counts = [floor(value) for value in scaled]
    for index in sorted(range(len(counts)), key=lambda i: (-(scaled[i]-counts[i]), i))[:count-sum(counts)]:
        counts[index] += 1
    return counts


def staged_sampling_probabilities(stage, blend=1.):
    """Population targets; only ascent assignments change after initialization."""
    if not 0 <= stage < len(SAMPLING_GROUP_SHARES) or not 0. <= blend <= 1.:
        raise ValueError(f"Unknown Stage9 sampling phase/blend: {stage}/{blend}")
    previous = SAMPLING_GROUP_SHARES[max(stage-1, 0)]
    group_shares = tuple(old+(new-old)*blend
                         for old,new in zip(previous, SAMPLING_GROUP_SHARES[stage]))
    return probabilities_for_shares(group_shares)


def probabilities_for_shares(group_shares, practice_share=NARROW_ASCENT_PRACTICE_SHARE):
    if len(group_shares) != 3 or min(group_shares) <= 0 or abs(sum(group_shares)-1.) > 1.e-9:
        raise ValueError('Invalid ascent population shares')
    probabilities = [0.] * TERRAIN_COLUMNS
    groups = (ASCENT_EASY_COLUMNS, ASCENT_MIDDLE_COLUMNS, ASCENT_HARD_COLUMNS)
    for columns, share in zip(groups, group_shares):
        for column in columns:
            probabilities[column] = ASCENT_FRACTION * share / len(columns)
    for column in DESCENT_COLUMNS:
        probabilities[column] = DESCENT_FRACTION / len(DESCENT_COLUMNS)
    for column in (*FLAT_COLUMNS, *GENERAL_COLUMNS):
        probabilities[column] = NON_STAIR_FRACTION / (len(FLAT_COLUMNS) + len(GENERAL_COLUMNS))
    if practice_share is not None:
        if not 0. < practice_share <= .10:
            raise ValueError('Narrow ascent practice share must be in (0,.10]')
        for column in NARROW_ASCENT_COLUMNS:
            target = ASCENT_FRACTION*practice_share/len(NARROW_ASCENT_COLUMNS)
            probabilities[column+1] += probabilities[column]-target
            probabilities[column] = target
    return tuple(probabilities)

if (ASCENT_REFERENCE_COLUMN != 8 or FLAT_COLUMNS != (9, 10, 11)
        or DESCENT_COLUMNS != tuple(range(12, 20))
        or GENERAL_COLUMNS != tuple(range(20, 28)) or TERRAIN_COLUMNS != 28):
    raise RuntimeError("Stage9 terrain column layout changed unexpectedly")
for _stage in range(len(SAMPLING_GROUP_SHARES)):
    if abs(sum(staged_sampling_probabilities(_stage)) - 1.) > 1.e-9:
        raise RuntimeError("Stage9 staged terrain probabilities must sum to one")


def task_label(direction, depth, width):
    return f"{direction}_d{int(round(depth * 100)):03d}_w{int(round(width * 100)):03d}"


def preserve_general_terrains(source_terrains):
    missing = [name for name in GENERAL_TERRAINS if name not in source_terrains]
    if missing:
        raise RuntimeError("Stage9 mixed-terrain source is missing official terrains: " + ", ".join(missing))
    return {name: deepcopy(source_terrains[name]) for name in GENERAL_TERRAINS}
