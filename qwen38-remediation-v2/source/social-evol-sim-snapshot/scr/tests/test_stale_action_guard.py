"""
Tests for the Phase 2 stale-action guard in _apply_decision().

Verifies that when a prey referenced by a Hunt action has already been
removed (e.g., by another agent's action in the same step), the ValueError
from update_checkpoint_from_actions is caught and the simulation continues
without crashing.
"""

from unittest.mock import patch

from scr.models.simulation.checkpoint import Checkpoint
from scr.models.agent.decision_result import AgentDecisionResult
from scr.models.agent.responses import Response
from scr.models.agent.actions import Hunt, Action
from scr.models.prompt_manager.messages import Messages
from scr.simulation.runner.simulation_step import _apply_decision


def _make_hunt_response(agent_id: str, prey_id: str) -> Response:
    """Create a valid Hunt response for testing."""
    return Response(
        agent_id=agent_id,
        thinking="I should hunt",
        short_term_plan="hunt prey",
        long_term_memory={},
        action=Action(root=Hunt(action_type="hunt", prey_id=prey_id, reason="test")),
    )


@patch("scr.simulation.runner.simulation_step.save_checkpoint")
class TestStaleActionGuard:
    """Phase 2 should not crash when a hunt targets already-removed prey."""

    def test_stale_hunt_prey_removed(self, mock_save_cp, tmp_path):
        """Hunt targeting prey that was removed mid-step is treated as DoNothing."""
        checkpoint = Checkpoint.initialize_from_config("configA_z8_easyHunting_visible")

        agent_id = checkpoint.metadata.execution_queue[0]
        agent = checkpoint.social_environment.get_agent_by_id(agent_id)
        prey = checkpoint.physical_environment.prey_animals[0]
        hp_before = agent.state.hp

        response = _make_hunt_response(agent_id, prey.id)

        # Remove the prey BEFORE applying the decision (simulates another agent killing it)
        checkpoint.physical_environment.prey_animals.remove(prey)

        result = AgentDecisionResult(
            agent_id=agent_id,
            response=response,
            messages=Messages(),
            success=True,
        )

        # Should NOT raise — the guard catches the ValueError
        _apply_decision(checkpoint, result, str(tmp_path))

        # Agent is still alive (only resistance damage at most, but prey was missing
        # so hunt() should never have been reached — guard catches ValueError first)
        assert agent.is_alive()
        # Response was recorded in history
        assert agent.response_history[-1] is response

    def test_normal_hunt_prey_exists(self, mock_save_cp, tmp_path):
        """Sanity check: hunt with existing prey applies normally."""
        checkpoint = Checkpoint.initialize_from_config("configA_z8_easyHunting_visible")

        agent_id = checkpoint.metadata.execution_queue[0]
        agent = checkpoint.social_environment.get_agent_by_id(agent_id)
        prey = checkpoint.physical_environment.prey_animals[0]
        prey_hp_before = prey.hp

        response = _make_hunt_response(agent_id, prey.id)
        result = AgentDecisionResult(
            agent_id=agent_id,
            response=response,
            messages=Messages(),
            success=True,
        )

        _apply_decision(checkpoint, result, str(tmp_path))

        # Agent is still alive (high HP config)
        assert agent.is_alive()
        # Response was recorded
        assert agent.response_history[-1] is response
        # Hunt had some effect: either prey took damage or was removed
        prey_still_present = any(
            p.id == prey.id for p in checkpoint.physical_environment.prey_animals
        )
        if prey_still_present:
            updated_prey = next(
                p for p in checkpoint.physical_environment.prey_animals if p.id == prey.id
            )
            assert updated_prey.hp <= prey_hp_before
        # else prey was killed and removed — also valid
