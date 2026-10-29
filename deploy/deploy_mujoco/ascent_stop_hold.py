"""External ascent-only zero-command hold; no learned parameters or actor inputs."""
import math

import numpy as np

VERSION = "ascent_stop_hold_20261028_v1"


class AscentStopHold:
    def __init__(self, default_position, action_scale, enabled=True):
        self.default = np.array(default_position, copy=True)
        self.scale = np.array(action_scale, copy=True)
        self.enabled = enabled
        self.reset()

    def reset(self):
        self.active = False
        self.legs_latched = False
        self.last_time = None
        self.release_start = None
        self.target_xy = np.zeros(2)
        self.target_yaw = 0.
        self.leg_target = np.zeros(12)
        self.wheel_reference = np.zeros(4)
        self.support_count = 0
        self.info = self._info()

    def _info(self, phase="inactive", error=0., yaw_error=0., blend=0.):
        return dict(
            ascent_stop_hold_active=self.active,
            ascent_stop_hold_phase=phase,
            ascent_stop_hold_blend=blend,
            ascent_stop_hold_position_error_m=float(error),
            ascent_stop_hold_yaw_error_rad=float(yaw_error),
            ascent_stop_hold_held_legs=4 if self.active and self.legs_latched else 0,
            ascent_stop_hold_support_wheels=self.support_count if self.active else 0,
            ascent_stop_hold_version=VERSION,
        )

    def update(self, now, requested, action, position, velocity, yaw, yaw_rate,
               joints, joint_velocity, support, wheel_lateral, eligible,
               hold_effort=None):
        """Brake first, then latch a stance with a feasible static load solution.

        The caller checks actual support, base speed and joint speed before
        supplying hold_effort. Do not wait for the zero-command policy to
        settle completely: that policy can keep retracting the supporting legs.
        Manual turns and reverse requests bypass this controller immediately.
        """
        action = np.asarray(action, dtype=float).copy()
        if self.last_time is not None and now < self.last_time:
            self.reset()
        dt = .02 if self.last_time is None else np.clip(now-self.last_time, 0., .04)
        self.last_time = now
        self.support_count = int(np.count_nonzero(support))
        zero = np.max(np.abs(requested)) <= 1.e-6
        straight_forward = requested[0] > .1 and np.max(np.abs(requested[1:])) <= 1.e-6
        if not self.enabled or not eligible or (not zero and not straight_forward):
            self.active = False
            self.release_start = None
            self.legs_latched = False
            self.info = self._info()
            return action
        if not self.active and not zero:
            self.info = self._info()
            return action
        if zero and (not self.active or self.release_start is not None):
            self.active = True
            self.release_start = None
            self.legs_latched = False
            self.target_xy = np.array(position, copy=True)
            self.target_yaw = yaw
            self.wheel_reference = joints[12:].copy()
        if zero and not self.legs_latched:
            self.wheel_reference = joints[12:].copy()
            self.target_xy = np.array(position, copy=True)
            if hold_effort is not None:
                self.legs_latched = True
                # Retain gravitational support through the existing 80 Nm/rad
                # leg PD. Measured dynamic effort is unsuitable as a preload.
                self.leg_target = joints[:12] + np.clip(hold_effort[:12], -76.4, 76.4)/80.
        error_xy = np.asarray(position)-self.target_xy
        direction = np.array([math.cos(self.target_yaw), math.sin(self.target_yaw)])
        longitudinal = float(error_xy@direction)
        yaw_error = math.atan2(math.sin(yaw-self.target_yaw), math.cos(yaw-self.target_yaw))
        if zero:
            correction = np.clip(-.5*longitudinal-.8*float(np.asarray(velocity)@direction), -.05, .05)
            turn = np.clip(-1.5*yaw_error-.5*yaw_rate, -.15, .15)
            # All M20 wheel axes point along local -Y: positive joint rotation
            # rolls the chassis backward along body X.
            if self.legs_latched:
                self.wheel_reference -= (correction-turn*np.asarray(wheel_lateral))/.09*dt
            blend = 1.
        else:
            if self.release_start is None:
                self.release_start = now
            blend = max(0., 1.-(now-self.release_start)/.12)
        held_action = action.copy()
        if self.legs_latched:
            held_action[:12] = (self.leg_target-self.default[:12])/self.scale[:12]
        # Encode the wheel servo through the existing velocity PD. Torque is
        # still limited to21.6 Nm by apply_pd; no added world force is used.
        torque = np.clip(12.*(self.wheel_reference-joints[12:])-1.2*joint_velocity[12:], -21.6, 21.6)
        held_action[12:] = (joint_velocity[12:]+torque/.6)/self.scale[12:]
        result = action+blend*(held_action-action)
        phase = ("hold" if self.legs_latched else "brake") if zero else "release"
        self.info = self._info(phase, float(np.linalg.norm(error_xy)), yaw_error, blend)
        if blend == 0.:
            self.active = False
            self.release_start = None
            self.legs_latched = False
            self.info = self._info()
        return np.clip(result, -100., 100.)
