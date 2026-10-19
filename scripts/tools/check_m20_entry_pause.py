#!/usr/bin/env python3
"""CPU state-machine checks for production first-tread command practice.

The parent fixture follows IsaacLab CommandTerm's metrics/resample/update order.
It substitutes command sampling and simulator bodies, not the teacher methods.
"""
from __future__ import annotations

import ast
import json
from types import SimpleNamespace

import torch

from check_m20_curriculum import SOURCE, production_namespace
from check_stair_rewards import Scene


class ParentCommand:
    def __init__(self, cfg, env):
        self.cfg, self._env, self.robot = cfg, env, env.scene["robot"]
        self.num_envs, self.device = env.num_envs, env.device
        self.vel_command_b = torch.tensor([[.6, 0., 0.]])
        self.time_left = torch.tensor([10.])
        self.is_standing_env = torch.zeros(1, dtype=torch.bool)
        self.is_heading_env = torch.zeros(1, dtype=torch.bool)
        self.metrics = {}
        self._metric_step_counter = torch.zeros(1)

    def _update_metrics(self):
        self._metric_step_counter += 1

    def _resample_command(self, ids):
        self.vel_command_b[ids] = torch.tensor([[-.5, 0., 0.]])
        self.is_standing_env[ids] = False
        self.is_heading_env[ids] = False

    def _resample(self, ids):
        self.time_left[ids] = 10.
        self._resample_command(ids)

    def _update_command(self):
        self.vel_command_b[self.is_standing_env] = 0.

    def compute(self, dt):
        self._update_metrics()
        self.time_left -= dt
        ids = (self.time_left <= 0).nonzero(as_tuple=False).flatten()
        if ids.numel():
            self._resample(ids)
        self._update_command()

    def reset(self, ids):
        extras = {key: float(value[ids].mean()) for key, value in self.metrics.items()}
        for value in self.metrics.values():
            value[ids] = 0.
        self._metric_step_counter[ids] = 0
        self._resample(ids)
        return extras


def fixture(probability=1.):
    ns = production_namespace()
    ns["UniformThresholdVelocityCommand"] = ParentCommand
    ns["TUNING"]["descent_entry_pause_probability"] = probability
    ns["TUNING"]["stair_zero_transition_probability"] = 0.
    ns["TUNING"]["stair_straight_practice_probability"] = 0.
    ns["_stair_activity_mask"] = lambda env: torch.ones(1, dtype=torch.bool)
    ns["_flat_surface_mask"] = lambda *args: torch.zeros(1, dtype=torch.bool)
    tree = ast.parse(SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "StairTeacherVelocityCommand")
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(SOURCE), "exec"), ns)
    lookup_count = {"robot": 0, "contact": 0}

    def find(which):
        def lookup(names, preserve_order):
            assert names == [f"{leg}_wheel" for leg in ("fl", "fr", "hl", "hr")]
            assert preserve_order
            lookup_count[which] += 1
            return list(range(4)), names
        return lookup

    robot = SimpleNamespace(find_bodies=find("robot"), data=SimpleNamespace(
        root_quat_w=torch.tensor([[1., 0, 0, 0]]),
        body_pos_w=torch.tensor([[[.1, .2, .94], [.1, -.2, .94], [-.5, .2, 1.09], [-.5, -.2, 1.09]]])))
    contacts = SimpleNamespace(find_bodies=find("contact"), data=SimpleNamespace(
        net_forces_w=torch.tensor([[[0., 0., 100.]] * 4]), current_contact_time=torch.ones(1, 4) * .1))
    env = SimpleNamespace(num_envs=1, device="cpu", common_step_counter=10, step_dt=.02,
                          episode_length_buf=torch.tensor([10]), scene=Scene(robot=robot, contact_forces=contacts),
                          _m20_stair_step_state=dict(active=torch.tensor([True]), up=torch.tensor([False]),
                              front_reached=torch.tensor([[True, True]]), target_z=torch.tensor([.85]), source_z=torch.tensor([1.])),
                          _m20_descent_entry_state=dict(active=torch.tensor([True]), count=torch.tensor([1]), top=torch.tensor([1.])))
    command = ns["StairTeacherVelocityCommand"](SimpleNamespace(resampling_time_range=(10., 10.)), env)
    return command, env, lookup_count


def tick(command, env):
    env.common_step_counter += 1
    env.episode_length_buf += 1
    command.compute(env.step_dt)


def main():
    torch.manual_seed(42)
    command, env, lookups = fixture()
    tick(command, env)
    assert command._entry_pause_pending.item() and command.vel_command_b.abs().sum() == 0
    assert 2 <= command.time_left.item() <= 3
    assert command.metrics["descent_entry_pause_events"].item() == 1
    assert command.metrics["descent_entry_pause_opportunities"].item() == 1
    elapsed = 0.
    while command._entry_pause_pending.item():
        tick(command, env)
        elapsed += env.step_dt
        assert elapsed < 3.1
    assert 2 <= elapsed <= 3.02
    assert .4 <= command.vel_command_b[0, 0].item() <= .7
    assert command.metrics["descent_entry_restart_events"].item() == 1
    for _ in range(10):
        tick(command, env)
    assert command.metrics["descent_entry_pause_events"].item() == 1
    assert lookups == {"robot": 1, "contact": 1}
    assert command.metrics["descent_entry_pause_time_fraction"].item() > 0

    skipped, skip_env, _ = fixture(probability=0.)
    for _ in range(10):
        tick(skipped, skip_env)
    assert skipped.metrics["descent_entry_pause_opportunities"].item() == 1
    assert skipped.metrics["descent_entry_pause_events"].item() == 0

    def reject(mutator):
        cmd, world, _ = fixture()
        mutator(cmd, world)
        tick(cmd, world)
        assert not cmd._entry_pause_pending.item()
        assert cmd.metrics["descent_entry_pause_opportunities"].item() == 0

    reject(lambda cmd, e: e.scene["contact_forces"].data.current_contact_time[0, 0].fill_(.02))
    reject(lambda cmd, e: e.scene["contact_forces"].data.net_forces_w[0, 2].zero_())
    reject(lambda cmd, e: e.scene["robot"].data.body_pos_w[0, 0, 2].fill_(.79))  # Front already on second tread.
    reject(lambda cmd, e: e.scene["robot"].data.body_pos_w[0, 2, 2].fill_(.94))  # Rear has departed platform.
    reject(lambda cmd, e: e._m20_stair_step_state["front_reached"][0, 1].fill_(False))
    reject(lambda cmd, e: cmd.vel_command_b.copy_(torch.tensor([[-.5, 0., 0.]])))
    reject(lambda cmd, e: cmd.vel_command_b.copy_(torch.tensor([[.5, 0., .5]])))
    reject(lambda cmd, e: e._m20_descent_entry_state["top"].fill_(1.2))

    # An ordinary early resample cancels the pause and preserves normal sampling.
    early, early_env, _ = fixture()
    tick(early, early_env)
    early._resample(torch.tensor([0]))
    assert not early._entry_pause_pending.item() and early.vel_command_b[0, 0] == -.5
    assert early.metrics["descent_entry_restart_events"].item() == 0
    # New explicit reverse/turn requests bypass the inherited standing mask.
    for request in ((-.4, 0., 0.), (0., 0., .6)):
        override, override_env, _ = fixture()
        tick(override, override_env)
        override.vel_command_b[:] = torch.tensor([request])
        tick(override, override_env)
        assert not override._entry_pause_pending.item()
        assert torch.allclose(override.vel_command_b, torch.tensor([request]))
        assert override.metrics["descent_entry_restart_events"].item() == 0
    # Reset cannot execute a pending restart, even when its timer is due.
    reset, reset_env, _ = fixture()
    tick(reset, reset_env)
    reset._entry_pause_timer_expired.fill_(True)
    extras = reset.reset(torch.tensor([0]))
    assert not reset._entry_pause_pending.item() and reset._entry_pause_decided_token.item() == -1
    assert reset.vel_command_b[0, 0] == -.5
    assert extras["descent_entry_pause_events"] == 1
    assert 0 <= extras["descent_entry_pause_time_fraction"] <= 1
    print(json.dumps(dict(first_tread_geometry_and_contacts_required=True,
                          selected_pause_then_forward_restart_elapsed_s=elapsed,
                          single_decision_even_when_not_selected=True,
                          no_repeat_same_physical_entry=True, wheel_body_lookups_cached=True,
                          early_resample_keeps_general_command=True, turn_reverse_cancel_pause=True,
                          reset_clears_pause_and_token=True, explicit_event_and_time_metrics=True), indent=2))


if __name__ == "__main__":
    main()
