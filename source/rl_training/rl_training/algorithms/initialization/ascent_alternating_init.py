"""Stage6 initialization: preserve inherited motion, train the sequence branch."""
from .ascent_adaptive_init import validate_source as adaptive_validate
from .ascent_adaptive_init import load_initialized_actor, load_initialized_critic, attach_curriculum_checkpoint


def validate_source(path, preserve=None):
    return adaptive_validate(path, preserve,
        accepted_profiles=('ascent_entry_v1_stage4', 'ascent_entry_v1_stage5', 'ascent_entry_v1_stage6'))
