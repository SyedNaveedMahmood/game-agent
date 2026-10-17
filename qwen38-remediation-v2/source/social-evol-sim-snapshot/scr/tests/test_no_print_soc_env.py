"""Verify soc_env_manager functions log to logger instead of print()."""

from unittest.mock import MagicMock, patch

from scr.simulation.env_manager.soc_env_manager import (
    update_agent_states,
    update_execution_queue,
)


def _make_mock_agent(agent_id="agent_1", age=5, hp=10, alive=True):
    agent = MagicMock()
    agent.id = agent_id
    agent.is_alive.return_value = alive
    agent.state.age = age
    agent.state.hp = hp
    return agent


def _make_mock_checkpoint(agents, current_step=0, comm_steps=0, max_age=20):
    cp = MagicMock()
    cp.social_environment.agents = agents
    cp.metadata.current_time_step = current_step
    cp.metadata.execution_queue = [a.id for a in agents]
    cp.metadata.current_agent_index = 0
    cp.configuration.world.communication_and_sharing_steps = comm_steps
    cp.configuration.agent.age.max = max_age
    return cp


class TestUpdateAgentStatesNoPrint:
    def test_no_stdout(self, capsys):
        """Agent state updates must go to logger, not stdout."""
        agent = _make_mock_agent()
        cp = _make_mock_checkpoint([agent])

        with patch("scr.simulation.env_manager.soc_env_manager.logger") as mock_logger:
            update_agent_states(cp)

        assert capsys.readouterr().out == ""
        mock_logger.debug.assert_called()

    def test_log_message_contains_agent_id(self):
        """Logger call should mention the agent id."""
        agent = _make_mock_agent(agent_id="agent_42")
        cp = _make_mock_checkpoint([agent])

        with patch("scr.simulation.env_manager.soc_env_manager.logger") as mock_logger:
            update_agent_states(cp)

        log_msg = mock_logger.debug.call_args[0][0]
        assert "agent_42" in log_msg


class TestUpdateExecutionQueueNoPrint:
    def test_no_stdout(self, capsys):
        """Execution queue update must go to logger, not stdout."""
        agent = _make_mock_agent()
        cp = _make_mock_checkpoint([agent])

        with patch("scr.simulation.env_manager.soc_env_manager.logger") as mock_logger:
            update_execution_queue(cp)

        assert capsys.readouterr().out == ""
        mock_logger.debug.assert_called()

    def test_log_message_contains_queue(self):
        """Logger call should contain the execution queue."""
        agent = _make_mock_agent(agent_id="agent_7")
        cp = _make_mock_checkpoint([agent])

        with patch("scr.simulation.env_manager.soc_env_manager.logger") as mock_logger:
            update_execution_queue(cp)

        log_msg = mock_logger.debug.call_args[0][0]
        assert "agent_7" in log_msg
