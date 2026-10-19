"""Command-independent terrain identity and stop exposure diagnostics.

No reward, observation input, command or actuator is modified here. Stair
identity describes height along body +X, independently of the sign of vx.
"""
from __future__ import annotations

import math
import torch


TERRAINS = ("flat", "stair_ascent", "stair_descent", "rough", "stair_unknown")
DIRECTIONS = ("after_forward", "after_reverse", "other")


def classify_terrain(env, scan, flat, stairs):
    """Use signed nearby geometry, never the command-gated up/down reward gates.

    Conflicting equally near edges stay unknown. A recent identity may bridge
    a missing scan only on nonflat stair support, within 1s/0.6m/45deg. Reset,
    clean flat terrain, turning and a contradictory visible edge invalidate it.
    """
    x = scan["ascending_edges_x"]
    delta = scan["ascending_edges_upper_z"] - scan["ascending_edges_lower_z"]
    nearby = scan["edge_geometry"] & (x >= -.45) & (x <= .70)
    distance = x.abs().expand_as(delta)
    up_distance = torch.where(nearby & (delta > 0), distance, torch.inf).amin(-1)
    down_distance = torch.where(nearby & (delta < 0), distance, torch.inf).amin(-1)
    up = torch.isfinite(up_distance) & (up_distance + .025 < down_distance)
    down = torch.isfinite(down_distance) & (down_distance + .025 < up_distance)
    visible = nearby.any(-1)
    # A signed discontinuity is stair evidence even at zero/reverse commands.
    stairs = stairs | visible
    robot = env.scene["robot"].data
    xy, q = robot.root_pos_w[:, :2], robot.root_quat_w
    yaw = torch.atan2(2*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]),
                      1-2*(q[:, 2].square()+q[:, 3].square()))
    step = int(env.common_step_counter)
    memory = getattr(env, "_m20_stop_terrain_identity", None)
    if memory is None:
        memory = dict(kind=torch.full_like(env.episode_length_buf, 4),
                      step=torch.full_like(env.episode_length_buf, -100000),
                      xy=xy.clone(), yaw=yaw.clone())
        env._m20_stop_terrain_identity = memory
    reset = env.episode_length_buf <= 1
    angle = torch.atan2(torch.sin(yaw-memory["yaw"]), torch.cos(yaw-memory["yaw"])).abs()
    remembered = ((step-memory["step"])*env.step_dt <= 1.) & (
        torch.linalg.vector_norm(xy-memory["xy"], dim=-1) <= .6) & (angle <= torch.pi/4)
    remembered &= ~reset & stairs & ~flat & ~visible
    kind = torch.full_like(env.episode_length_buf, 3)
    kind[flat & ~stairs] = 0
    kind[stairs] = 4
    kind[remembered] = memory["kind"][remembered]
    kind[up] = 1
    kind[down] = 2
    fresh = up | down
    memory["kind"][fresh] = kind[fresh]
    memory["step"][fresh] = step
    memory["xy"][fresh] = xy[fresh]
    memory["yaw"][fresh] = yaw[fresh]
    invalid = ((reset & ~fresh) | (flat & ~stairs) | (visible & ~fresh) | (angle > torch.pi/4)) & ~fresh
    memory["step"][invalid] = -100000
    return kind


class StopCoverage:
    """Per-episode counts separated by terrain and preceding raw command.

    Existing stop metrics retain the entry terrain. These additional metrics
    distinguish time commanded to stop from time actually remaining on that
    terrain. Restart success means >=0.2m progress along a new translation
    request within 3s; it is not a full gait/safety acceptance test.
    """
    fields = ("motion_samples", "opportunities", "entries", "measured_samples",
              "on_terrain_samples", "observed_4s_holds", "ended_holds",
              "ended_without_quiet", "restart_requests", "restart_progress", "restart_timeouts", "restart_cancelled")

    def __init__(self, env):
        n, device = env.num_envs, env.device
        self.dt = env.step_dt
        self.covered_steps = math.ceil(4. / self.dt)
        self.groups = tuple(f"{terrain}/{direction}" for terrain in TERRAINS for direction in DIRECTIONS)
        self.stats = {field: torch.zeros(n, len(self.groups), device=device) for field in self.fields}
        self.last_command = torch.zeros(n, 3, device=device)
        self.motion_group = torch.full((n,), -1, dtype=torch.long, device=device)
        self.hold_group = torch.zeros(n, dtype=torch.long, device=device)
        self.was_zero = torch.zeros(n, dtype=torch.bool, device=device)
        self.on_terrain_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.quiet_seen = torch.zeros(n, dtype=torch.bool, device=device)
        self.restart_active = torch.zeros(n, dtype=torch.bool, device=device)
        self.restart_group = self.hold_group.clone()
        self.restart_xy = torch.zeros(n, 2, device=device)
        self.restart_direction = torch.zeros(n, 2, device=device)
        self.restart_age = torch.zeros(n, device=device)
        self.row = torch.arange(n, device=device)

    @staticmethod
    def direction(command):
        straight = (command[:, 1].abs() <= .1) & (command[:, 2].abs() <= .1)
        direction = torch.full_like(command[:, 0], 2, dtype=torch.long)
        direction[straight & (command[:, 0] > .1)] = 0
        direction[straight & (command[:, 0] < -.1)] = 1
        return direction

    def _add(self, field, group, mask):
        self.stats[field][self.row, group] += mask.float()

    def update(self, env, command, kind, entry_kind, measured, quiet):
        reset = env.episode_length_buf <= 1
        for value in self.stats.values():
            value[reset] = 0
        self.last_command[reset] = 0
        self.was_zero[reset] = False
        self.motion_group[reset] = -1
        self.restart_active[reset] = False
        zero = command.abs().amax(-1) <= 1.e-4
        entering, leaving = zero & ~self.was_zero, ~zero & self.was_zero
        group = kind*3 + self.direction(command)
        motion = (~zero & (self.direction(command) < 2) & ~reset)
        self._add("motion_samples", group, motion)
        self._add("opportunities", group, motion & (group != self.motion_group))
        self.motion_group = torch.where(motion, group, -1)
        self._add("ended_holds", self.hold_group, leaving & ~reset)
        self._add("ended_without_quiet", self.hold_group, leaving & ~reset & ~self.quiet_seen)
        xy = env.scene["robot"].data.root_pos_w[:, :2]
        restart = leaving & ~reset & (torch.linalg.vector_norm(command[:, :2], dim=-1) > .1)
        self._add("restart_requests", self.hold_group, restart)
        self._add("restart_cancelled", self.restart_group, self.restart_active & zero & ~reset)
        self.restart_active[zero | reset] = False
        self.restart_active[restart] = True
        self.restart_group[restart] = self.hold_group[restart]
        self.restart_xy[restart] = xy[restart]
        self.restart_age[restart] = 0
        q = env.scene["robot"].data.root_quat_w
        yaw = torch.atan2(2*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]),
                          1-2*(q[:, 2].square()+q[:, 3].square()))
        c, s = torch.cos(yaw), torch.sin(yaw)
        world = torch.stack((c*command[:, 0]-s*command[:, 1], s*command[:, 0]+c*command[:, 1]), -1)
        world /= torch.linalg.vector_norm(world, dim=-1, keepdim=True).clamp_min(1.e-6)
        self.restart_direction[restart] = world[restart]
        self.restart_age += self.restart_active*self.dt
        progress = ((xy-self.restart_xy)*self.restart_direction).sum(-1)
        reached = self.restart_active & (progress >= .2) & (self.restart_age <= 3.+1.e-6)
        timeout = self.restart_active & (self.restart_age > 3.+1.e-6)
        self._add("restart_progress", self.restart_group, reached)
        self._add("restart_timeouts", self.restart_group, timeout)
        self.restart_active[reached | timeout] = False
        self.hold_group[entering] = (entry_kind*3+self.direction(self.last_command))[entering]
        self.on_terrain_steps[entering] = 0
        self.quiet_seen[entering] = False
        self._add("entries", self.hold_group, entering)
        self._add("measured_samples", self.hold_group, measured)
        on_terrain = measured & (kind == entry_kind)
        before = self.on_terrain_steps.clone()
        self.on_terrain_steps += on_terrain.long()
        covered = (before < self.covered_steps) & (self.on_terrain_steps >= self.covered_steps)
        self._add("on_terrain_samples", self.hold_group, on_terrain)
        self._add("observed_4s_holds", self.hold_group, covered & zero)
        self.quiet_seen |= zero & quiet
        self.was_zero = zero.clone()
        self.last_command = command.clone()

    def metrics(self, env_ids):
        output = {}
        for i, name in enumerate(self.groups):
            for field, value in self.stats.items():
                output[f"coverage/{name}/{field}"] = value[env_ids, i].sum()
            # Curriculum is called before an episode reset. Count the active
            # episode-ending hold here, without inventing a command restart.
            output[f"coverage/{name}/active_at_episode_end"] = (
                self.was_zero[env_ids] & (self.hold_group[env_ids] == i)).float().sum()
            output[f"coverage/{name}/restart_pending_at_episode_end"] = (
                self.restart_active[env_ids] & (self.restart_group[env_ids] == i)).float().sum()
            for source, label in (("motion_samples", "motion_s"), ("measured_samples", "measured_s"),
                                  ("on_terrain_samples", "on_terrain_s")):
                output[f"coverage/{name}/{label}"] = self.stats[source][env_ids, i].sum()*self.dt
        return output
