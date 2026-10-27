"""Population quotas and minimal reset-time ascent transfers, without Isaac."""
import torch
from . import stair_path_layout as layout


def probabilities_at(iteration, readiness=None):
    # No wall-clock/iteration fallback can silently bypass the readiness gate.
    return (readiness.probabilities(iteration) if readiness is not None
            else layout.probabilities_for_shares(layout.READINESS_GROUP_SHARES[0]))


def initial_assignment(count, device):
    counts = layout.integer_sample_counts(probabilities_at(0), count)
    columns = torch.repeat_interleave(torch.arange(layout.TERRAIN_COLUMNS, device=device),
                                    torch.tensor(counts, device=device))
    return columns[torch.randperm(count, device=device)]


class AscentPopulation:
    """Derived quota cache, invalidated whenever terrain assignments change.

    Rows/prefixes can change freely: only column assignments affect the quota.
    Keep the tiny histogram on the host to avoid device syncs on stable resets.
    This cache is rebuilt after restore and never enters checkpoint state.
    """
    def __init__(self, terrain_types):
        self.counts = torch.bincount(terrain_types, minlength=layout.TERRAIN_COLUMNS)
        self.current = self.counts.cpu().tolist()
        self.ascent_count = sum(self.current[:layout.ASCENT_COLUMNS])
        self._probabilities = None
        self._targets = None

    def targets(self, iteration, readiness=None):
        probabilities = probabilities_at(iteration, readiness)
        if probabilities != self._probabilities:
            self._targets = layout.integer_sample_counts(
                [p/layout.ASCENT_FRACTION for p in probabilities[:layout.ASCENT_COLUMNS]], self.ascent_count)
            self._probabilities = probabilities
        return self._targets

    def pending(self, targets):
        return sum(max(value-target, 0) for value,target in zip(self.current, targets))


def ascent_targets(terrain_types, iteration, readiness=None):
    population = AscentPopulation(terrain_types)
    return population.counts, population.targets(iteration, readiness)


def plan_ascent_moves(terrain_types, reset_ids, iteration, readiness=None, population=None):
    """Move only resetting members of surplus columns into current deficits.

    The fixed ascent pool never borrows descent/general environments. Repeated
    resets at an unchanged, satisfied target produce no assignment churn.
    """
    population = population if population is not None else AscentPopulation(terrain_types)
    targets = population.targets(iteration, readiness)
    current = population.current[:layout.ASCENT_COLUMNS]
    if not population.pending(targets):
        return reset_ids[:0], terrain_types[:0]
    deficit = [max(target-value, 0) for value,target in zip(current, targets)]
    candidates = []
    reset_types = terrain_types[reset_ids]
    for column, (value, target) in enumerate(zip(current, targets)):
        surplus = value-target
        if surplus <= 0:
            continue
        ids = reset_ids[reset_types == column]
        if len(ids):
            candidates.append(ids[torch.randperm(len(ids), device=ids.device)[:surplus]])
    if not candidates:
        return reset_ids[:0], terrain_types[:0]
    changed_ids = torch.cat(candidates)
    destinations = torch.repeat_interleave(
        torch.arange(layout.ASCENT_COLUMNS, device=terrain_types.device),
        torch.tensor(deficit, device=terrain_types.device))
    destinations = destinations[torch.randperm(len(destinations), device=destinations.device)[:len(changed_ids)]]
    return changed_ids, destinations


def reset_ascent_progress(state, ids):
    """A changed geometry gets its pinned prefix and fresh speed-balanced history."""
    state.skill[ids] = state.minimum_skill[ids]
    for name in ('good', 'bad', 'completed', 'window', 'trials', 'cursor', 'speed_bucket'):
        getattr(state, name)[ids] = 0
    if hasattr(state,'reset_geometry'):
        state.reset_geometry(ids)
