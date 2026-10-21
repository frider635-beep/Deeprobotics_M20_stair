# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""M20 reinforcement-learning tasks and training support modules."""

# Resolve historical checkpoint imports without leaving wrappers at the root.
from .utils.legacy_imports import install as _install_legacy_imports

_install_legacy_imports()

# Register Gym environments.
from .tasks import *
