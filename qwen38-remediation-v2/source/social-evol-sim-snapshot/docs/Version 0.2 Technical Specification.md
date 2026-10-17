# **Hunting System**

## **1. Resource Mechanics**
### **1.1 Prey Animals**
- **Standard Prey**
  - HP: 4
  - Meat Yield: 2 units (calculated as max_hp / meat_unit_hp = 4 / 2)
  - Attack: 1 HP damage on failed hunt
  - Nutrition: 20 HP per meat unit when consumed

### **1.2 Plants**
- **Plant Resources**
  - Capacity: 5 plants per cell
  - Respawn Delay: 10 turns
  - Nutrition: 10 HP per plant
  - Collection: Up to 2 plants per action (max_collect_quantity)

### **1.3 Resource Comparison**
| Resource Type | Risk | Reward | Collection Limit | Nutrition |
|--------------|------|--------|------------------|-----------|
| Plants       | None | Low    | 2 per action     | 10 HP     |
| Prey         | High | High   | 2 meat units     | 20 HP     |

## **2. Hunting Mechanics**
### **2.1 Hunt Action**
- **Success Chance**: 
  - Base success rate: attack_power/10
  - Maximum success rate: 90% (capped)
  - Formula: min(attack_power/10, 0.9)
- **On Success**:
  - Agent deals damage equal to their attack power
  - If prey HP reaches 0, it is killed and the agent gets meat
  - If prey survives, it remains in the environment
  - Only the agent that deals the killing blow gets the meat
- **On Failure**:
  - Prey counter-attacks for 1 HP damage
  - No meat obtained

### **2.2 Prey Spawning**
- **Initial Quantity**: Configurable (default: 3)
- **Respawn Rate**: 10% chance per turn at empty locations
- **Location**: Random unoccupied cells

### **2.3 Inventory Management**
- **Meat Storage**: 
  - Meat units stored in inventory
  - Each unit restores 20 HP when consumed
  - Subject to inventory size limits
  - Only obtained by the agent that kills the prey

---

## **3. Hunting Phases**
### **3.1 Pre-Hunting**
1. **Detection**: Agents must locate prey in adjacent cells
2. **Preparation**: Ensure sufficient HP to absorb potential counter-attack damage
3. **Attack Power**: Consider agent's attack power for:
   - Increased hunt success rate (attack_power/10)
   - Higher damage output on successful hits

### **3.2 Hunting**
- **Risk Assessment**: Balance potential meat reward (20 HP) against risk of damage (1 HP)
- **Inventory Check**: Verify space for potential meat before hunting
- **Action Execution**: Single hunt action with:
  - Success chance based on attack power (attack_power/10)
  - Damage based on attack power
- **Kill Confirmation**: Only successful if prey HP reaches 0

### **3.3 Post-Hunt**
1. **Success with Kill**: Meat automatically added to inventory of killing agent
2. **Success without Kill**: Prey remains with reduced HP
3. **Failure**: Agent takes damage, prey remains alive
4. **Cleanup**: Dead prey removed from environment

---

## **4. Integrated Simulation Flow**
**Action Processing**:
1. Hunt action validation
2. Success/failure determination based on attack power (attack_power/10)
3. Damage application based on attack power
4. Kill confirmation
5. Meat distribution to killing agent
6. Dead prey removal

**Environment Updates**:
- Prey spawn probability: 10% per turn at empty locations
- No meat spoilage implemented

---

## **5. Balance Parameters**
- **Prey Stats**:
  - HP: 4
  - Attack: 1
  - Meat Yield: 2 units
  - Nutrition: 20 HP per unit

- **Agent Considerations**:
  - Inventory size limits apply to meat storage
  - Meat provides 2x nutrition of plants (20 HP vs 10 HP)
  - Failed hunts cost 1 HP
  - Attack power affects:
    - Hunt success rate (attack_power/10, capped at 90%)
    - Damage dealt to prey
  - Only killing agent receives meat reward

---

## **6. Moral vs Immoral Behavior**
### **6.1 Moral Agents**
- Coordinate hunting efforts
- Share meat with low-HP agents
- Prioritize sustainable hunting
- Protect other moral agents

### **6.2 Immoral Agents**
- Hunt aggressively
- Hoard meat resources
- May attack during hunts
- Use deception in communication

## **7. Configuration Options**
### **7.1 World Settings**
```json
"world": {
  "width": 16,        // Grid width
  "height": 16,       // Grid height
  "max_life_steps": 3 // Maximum simulation steps
}
```

### **7.2 Agent Settings**
```json
"agent": {
  "initial_count": 2,  // Number of agents at start
  "ratio": {
    "moral": 0,        // Ratio of moral agents (0-1)
    "immoral": 1       // Ratio of immoral agents (0-1)
  },
  "hp": {
    "initial": 45,     // Starting HP for agents
    "max": 60          // Maximum possible HP
  },
  "age": {
    "initial": 10      // Starting age for agents
  },
  "inventory": {
    "max_size": 10     // Maximum items in inventory
  },
  "reproduction": {
    "min_hp": 11,      // Minimum HP required to reproduce
    "hp_cost": 10,     // HP cost to reproduce
    "min_age": 4,      // Minimum age to reproduce
    "offspring_initial_hp": 3  // HP of new offspring
  },
  "attack": {
    "power": {
      "mean": 5,       // Mean attack power
      "sd": 2.5        // Standard deviation of attack power
    }
  },
  "max_collect_quantity": 2  // Maximum items collectible per action
}
```

### **7.3 Resource Settings**
```json
"resource": {
  "plant": {
    "initial_quantity": 0,    // Starting number of plants
    "capacity": 5,            // Maximum plants per cell
    "respawn_delay": 10,      // Turns until plant respawns
    "nutrition": 10           // HP restored per plant
  },
  "prey": {
    "initial_quantity": 3,    // Starting number of prey
    "hp": 4,                 // Prey health points
    "attack": 1,             // Damage on failed hunt
    "respawn_rate": 0.1,     // Chance to spawn per turn
    "meat_unit_hp": 2,       // HP per meat unit
    "nutrition": 20          // HP restored per meat unit
  }
}
```

### **7.4 Configuration Guidelines**
1. **World Size**
   - Larger worlds (width × height) allow for more prey and agents
   - Consider adjusting prey initial_quantity based on world size

2. **Agent Balance**
   - HP settings affect hunting risk tolerance
   - Inventory size affects resource hoarding capacity
   - Attack power affects both hunt success and damage output
   - Reproduction settings control population growth

3. **Resource Balance**
   - Plant vs prey nutrition ratio (10:20) affects resource preference
   - Prey respawn_rate (0.1) affects resource availability
   - Initial quantities affect early game dynamics

4. **Moral Ratio**
   - Adjust moral:immoral ratio to test different social dynamics
   - 0:1 for pure competition
   - 1:0 for pure cooperation
   - Mixed ratios for complex social interactions

## **8. Mathematical Modeling of Resource-Population Dynamics**

### **8.1 Resource Consumption Requirements**
Let \( h_t \) represent an agent's HP at time step \( t \), then:
\[ h_{t+1} = h_t - \Delta h + n \]
where:
- \( \Delta h = 1 \) (base HP loss per step)
- \( n \) is nutrition gained from resources

**Nutrition Requirements**:
- Plant-based diet: \( n_p = 10 \) HP per plant
- Meat-based diet: \( n_m = 20 \) HP per meat unit

### **8.2 Resource Production Model**
#### **Plant Production**
Let \( P_t \) be the number of plants at time \( t \), \( C = 5 \) be cell capacity, and \( D = 10 \) be respawn delay:
\[ P_{t+1} = \min(P_t + \frac{C}{D}, C) \]
\[ \text{Production Rate} = \frac{C}{D} = 0.5 \text{ plants per cell per step} \]

#### **Prey Production**
Let \( M_t \) be meat units at time \( t \), \( r = 0.1 \) be respawn rate, and \( y = 2 \) be meat yield:
\[ M_{t+1} = M_t + r \cdot y \]
\[ \text{Expected Meat Production} = r \cdot y = 0.2 \text{ units per step} \]

### **8.3 Population Sustainability Analysis**
#### **World Parameters**
- Grid size: \( G = 16 \times 16 = 256 \) cells
- Initial agents: \( A_0 = 2 \)
- Agent HP range: \( h \in [0, 60] \)
- Initial HP: \( h_0 = 45 \)
- Attack power range: \( a \in [1, 9] \)
- Attack power mean: \( \mu_a = 5 \)
- Attack power sd: \( \sigma_a = 2.5 \)

#### **Resource Requirements**
For \( n \) agents:
\[ \text{Plant Requirement} = n \cdot \frac{\Delta h}{n_p} = n \text{ plants per step} \]
\[ \text{Meat Requirement} = n \cdot \frac{\Delta h}{n_m} = \frac{n}{2} \text{ meat units per step} \]

#### **Resource Production Capacity**
\[ \text{Total Plant Production} = G \cdot \frac{C}{D} = 128 \text{ plants per step} \]
\[ \text{Total Meat Production} = r \cdot y = 0.2 \text{ units per step} \]

### **8.4 Population Growth Constraints**
#### **Reproduction Parameters**
- Minimum HP for reproduction: \( h_{\min} = 11 \)
- Reproduction cost: \( c_r = 10 \) HP
- Minimum age: \( a_{\min} = 4 \)
- Offspring HP: \( h_{\text{offspring}} = 3 \)

#### **Growth Function**
Let \( A_t \) be population at time \( t \):
\[ A_{t+1} = A_t + \sum_{i=1}^{A_t} R_i \]
where \( R_i \) is reproduction function for agent \( i \):
\[ R_i = \begin{cases} 
1 & \text{if } h_i > h_{\min} \text{ and } a_i \geq a_{\min} \\
0 & \text{otherwise}
\end{cases} \]

### **8.5 Equilibrium Analysis**
#### **Resource Equilibrium Points**
- Plant-based equilibrium:
\[ A_{\text{eq}}^{\text{plant}} = \frac{G \cdot C}{D \cdot \Delta h} \cdot n_p \approx 128 \text{ agents} \]

- Meat-based equilibrium:
\[ A_{\text{eq}}^{\text{meat}} = \frac{r \cdot y}{\Delta h} \cdot n_m \approx 0.4 \text{ agents} \]

#### **Population Dynamics**
The population growth rate \( \lambda \) is constrained by:
\[ \lambda \leq \min\left(\frac{G \cdot C}{D \cdot A_t}, \frac{r \cdot y \cdot n_m}{A_t \cdot \Delta h}\right) \]

### **8.6 Optimization Recommendations**
1. **Resource Balance**:
   - Prey quantity: \( M_0 \in [3, 5] \)
   - Plant respawn: \( D' = 5 \)
   - Meat unit HP: \( n_m' = 10 \)

2. **Population Settings**:
   - Initial agents: \( A_0' \in [4, 6] \)
   - Time steps: \( T \in [50, 100] \)
   - HP loss: \( \Delta h' = 0.5 \)

3. **World Parameters**:
   - Current grid: \( G = 256 \) cells
   - Scaling factor: \( s = \frac{A_t}{A_0} \)
   - Optimal grid: \( G' = G \cdot s \)
