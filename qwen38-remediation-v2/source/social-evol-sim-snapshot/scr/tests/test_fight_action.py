import unittest
from scr.models.simulation.checkpoint import Checkpoint
from scr.models.agent.agent import Agent
from scr.models.agent.actions import Fight
from scr.models.agent.agent import AgentState
from scr.simulation.act_manager.action_handler.fight import fight

class TestFightAction(unittest.TestCase):
    def setUp(self):
        # Create a checkpoint
        self.checkpoint = Checkpoint.initialize_from_config("configA_z8_easyHunting_visible")
        
        # Create two agents with different physical abilities
        self.agent1 = Agent(
            id="agent1",
            type="moral",
            state=AgentState(
                hp=10,
                physical_ability=8,
                phisical_ability_scaling={'slope': 2, 'intercept': 0.1}
            )
        )
        
        self.agent2 = Agent(
            id="agent2",
            type="immoral",
            state=AgentState(
                hp=10,
                physical_ability=5,
                phisical_ability_scaling={'slope': 2, 'intercept': 0.1}
            )
        )
        
        # Add agents to the checkpoint
        self.checkpoint.social_environment.agents = [self.agent1, self.agent2]
        
    def test_fight_success(self):
        # Create a fight action
        action = Fight(action_type="fight", reason="test", target_agent_id="agent2", fight_reason_label="retaliate_against_attack")
        
        # Execute the fight
        updated_checkpoint = fight(self.checkpoint, self.agent1, action)
        
        # Verify results
        self.assertLess(self.agent1.state.hp, 10)  # Agent1 should take resistance damage
        self.assertLessEqual(self.agent2.state.hp, 10)  # Agent2 might take damage
        
        # Check if the fight was recorded in observations
        self.assertTrue(any("agent1" in obs for obs in updated_checkpoint.observations))
        
    def test_fight_self(self):
        # Try to fight self
        action = Fight(action_type="fight", reason="test", target_agent_id="agent1", fight_reason_label="retaliate_against_attack")
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            fight(self.checkpoint, self.agent1, action)
            
    def test_fight_dead_agent(self):
        # Make agent2 dead
        self.agent2.state.hp = 0
        
        # Try to fight dead agent
        action = Fight(action_type="fight", reason="test", target_agent_id="agent2", fight_reason_label="retaliate_against_attack")
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            fight(self.checkpoint, self.agent1, action)
            
    def test_fight_nonexistent_agent(self):
        # Try to fight non-existent agent
        action = Fight(action_type="fight", reason="test", target_agent_id="nonexistent", fight_reason_label="retaliate_against_attack")
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            fight(self.checkpoint, self.agent1, action)

if __name__ == '__main__':
    unittest.main() 