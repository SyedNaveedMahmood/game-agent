"""Tests for physical environment updates."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from scr.models.environment.prey import PreyAnimal
from scr.simulation.env_manager.phy_env_manager import phy_env_step


def test_prey_spawn_observation_uses_current_prey_reward_fields():
    prey_animals = []

    def spawn_new_prey():
        prey_animals.append(
            PreyAnimal(
                id="prey_test",
                hp=12,
                max_hp=12,
                physical_ability=4,
                num_agents_to_kill=4,
            )
        )

    checkpoint = SimpleNamespace(
        metadata=SimpleNamespace(current_time_step=12),
        physical_environment=SimpleNamespace(
            resources=[],
            prey_animals=prey_animals,
            spawn_new_prey=spawn_new_prey,
        ),
        add_observation=MagicMock(),
    )

    phy_env_step(checkpoint)

    checkpoint.add_observation.assert_called_once()
    details = checkpoint.add_observation.call_args.kwargs["details"]
    assert "reward_hp=12" in details
    assert "num_agents_to_kill=4" in details
    assert "meat_units" not in details
