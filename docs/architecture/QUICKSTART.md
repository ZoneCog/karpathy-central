# Quick Start Guide

## Overview

This guide will help you get started with the Distributed Agentic Cognitive Grammar system. The system provides a unified framework for coordinating multiple AI subsystems through an intelligent agent network.

## Installation and Setup

### Prerequisites

- Python 3.8+
- Basic understanding of neural networks and distributed systems
- Access to existing neural architectures (ScriptBots, RecurrentJS, NeuralTalk, etc.)

### Quick Setup

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/ZoneCog/karpathy-central.git
   cd karpathy-central
   ```

2. **Install dependencies**:
   ```bash
   pip install asyncio numpy scipy
   # Add other dependencies as needed for specific integrations
   ```

3. **Run the basic system**:
   ```python
   python -m cognitive_grammar.orchestrator
   ```

## Basic Usage

### Starting the System

```python
import asyncio
from cognitive_grammar.orchestrator import CognitiveGrammarSystem, SystemConfiguration

async def start_system():
    # Create configuration
    config = SystemConfiguration()
    config.enable_memory_agent = True
    config.enable_task_manager = True
    config.enable_cognitive_agent = True
    config.enable_autonomy_agent = True
    
    # Create system
    system = CognitiveGrammarSystem(config)
    
    # Initialize
    await system.initialize()
    
    # Start processing
    await system.start()

# Run the system
asyncio.run(start_system())
```

### Submitting Goals

The system accepts high-level goals that are automatically decomposed into tasks:

```python
# Learning goal
await system.submit_goal(
    goal_name="learn_language_patterns",
    goal_description="Learn natural language patterns from text data",
    requirements=["language_processing", "pattern_recognition"],
    input_data={
        "dataset": "text_corpus.txt",
        "model_type": "transformer",
        "epochs": 10
    },
    priority=3  # High priority
)

# Reasoning goal
await system.submit_goal(
    goal_name="multimodal_reasoning",
    goal_description="Perform reasoning across visual and textual modalities",
    requirements=["visual_processing", "language_understanding", "reasoning"],
    input_data={
        "image_path": "example.jpg",
        "question": "What is happening in this image?",
        "context": "urban environment"
    },
    priority=2
)
```

### Querying Knowledge

```python
# Query the distributed knowledge base
result = await system.query_knowledge(
    query="neural network optimization techniques",
    domain="machine_learning"
)

# Query with specific parameters
result = await system.query_knowledge(
    query="attention mechanisms in transformers",
    domain="natural_language_processing"
)
```

### Monitoring System Status

```python
# Get comprehensive system status
status = system.get_system_status()

print(f"System uptime: {status['uptime']:.2f} seconds")
print(f"Active agents: {len(status['agents'])}")
print(f"Messages processed: {status['total_messages_processed']}")
print(f"Knowledge nodes: {status['hypergraph_stats']['total_nodes']}")
```

## Agent Interaction Examples

### Memory Agent Interaction

```python
from cognitive_grammar.agents.memory_agent import MemoryAgent
from cognitive_grammar.atomspace.hypergraph import Hypergraph

# Create hypergraph and memory agent
hg = Hypergraph("ExampleKnowledge")
memory_agent = MemoryAgent("memory_001", hg)

# Store a pattern
pattern_task = {
    'type': 'store_pattern',
    'pattern_data': {
        'input': [1, 0, 1, 0],
        'output': [0, 1],
        'accuracy': 0.95
    },
    'pattern_type': 'classification',
    'confidence': 0.9,
    'domain': 'binary_classification'
}

result = await memory_agent.process_task(pattern_task)
print(f"Pattern stored: {result}")

# Retrieve context
context_task = {
    'type': 'retrieve_context',
    'query': 'binary classification patterns',
    'context_size': 5,
    'domain': 'binary_classification'
}

context = await memory_agent.process_task(context_task)
print(f"Retrieved context: {len(context['result']['context_nodes'])} nodes")
```

### Cognitive Agent Interaction

```python
from cognitive_grammar.agents.cognitive_agent import CognitiveAgent

# Create cognitive agent with multiple backends
cognitive_agent = CognitiveAgent("cognitive_001", hg, ['scriptbots', 'recurrentjs'])

# Neural inference task
inference_task = {
    'type': 'neural_inference',
    'input_data': {
        'sequence': [[1, 0, 1], [0, 1, 0], [1, 1, 0]],
        'model_type': 'lstm'
    },
    'backend': 'recurrentjs'
}

result = await cognitive_agent.process_task(inference_task)
print(f"Inference result: {result}")

# Pattern recognition task
pattern_task = {
    'type': 'pattern_recognition',
    'data': {'features': [0.1, 0.5, 0.8, 0.2]},
    'pattern_type': 'classification',
    'threshold': 0.7
}

patterns = await cognitive_agent.process_task(pattern_task)
print(f"Similar patterns found: {len(patterns['result']['similar_patterns'])}")
```

### Task Manager Interaction

```python
from cognitive_grammar.agents.task_agent import TaskManagerAgent

# Create task manager
task_manager = TaskManagerAgent("task_mgr_001", hg)

# Submit a complex goal
goal_task = {
    'type': 'submit_goal',
    'name': 'image_captioning_pipeline',
    'description': 'Create an end-to-end image captioning system',
    'requirements': ['image_processing', 'language_generation'],
    'input_data': {
        'image_dataset': 'coco_images/',
        'caption_dataset': 'coco_captions.json',
        'model_architecture': 'cnn_lstm'
    },
    'priority': 3
}

result = await task_manager.process_task(goal_task)
print(f"Goal submitted with {result['result']['subtasks_created']} subtasks")
```

## Integration with Existing Systems

### ScriptBots Integration

```python
# The cognitive agent automatically integrates with ScriptBots
# when the scriptbots backend is enabled

cognitive_task = {
    'type': 'learning',
    'learning_type': 'evolutionary',
    'backend': 'scriptbots',
    'training_data': {
        'population_size': 100,
        'generations': 20,
        'fitness_function': 'survival_time'
    }
}

# This will use ScriptBots' evolutionary algorithms
result = await cognitive_agent.process_task(cognitive_task)
```

### RecurrentJS Integration

```python
# Use RecurrentJS for sequence modeling
sequence_task = {
    'type': 'neural_inference',
    'input_data': {
        'sequence': text_to_sequence("hello world"),
        'model_type': 'lstm'
    },
    'backend': 'recurrentjs'
}

result = await cognitive_agent.process_task(sequence_task)
```

### NeuralTalk Integration

```python
# Use NeuralTalk for image captioning
caption_task = {
    'type': 'neural_inference',
    'input_data': {
        'image_path': 'example_image.jpg'
    },
    'backend': 'neuraltalk'
}

caption = await cognitive_agent.process_task(caption_task)
print(f"Generated caption: {caption['result']['caption']}")
```

## Advanced Features

### Custom Agent Creation

```python
from cognitive_grammar.agents.base_agent import BaseAgent

class CustomAgent(BaseAgent):
    def __init__(self, agent_id, hypergraph):
        capabilities = ['custom_processing', 'special_analysis']
        super().__init__(agent_id, "CustomAgent", hypergraph, capabilities)
    
    def _initialize(self):
        # Custom initialization
        self.custom_data = {}
        self.state = AgentState.ACTIVE
    
    async def process_task(self, task):
        # Custom task processing
        if task.get('type') == 'custom_analysis':
            return await self._custom_analysis(task)
        return {'error': 'Unknown task type'}
    
    async def _custom_analysis(self, task):
        # Implement custom analysis logic
        return {'result': 'custom_analysis_complete'}
    
    async def _process_cycle(self):
        # Custom processing cycle
        pass
    
    def get_status(self):
        base_status = self.get_base_status()
        base_status['custom_data_size'] = len(self.custom_data)
        return base_status

# Register and use custom agent
custom_agent = CustomAgent("custom_001", hg)
await system._register_agent(custom_agent)
```

### Custom Routing Rules

```python
from cognitive_grammar.communication.message_bus import RoutingRule

# Create custom routing rule
def priority_boost_transform(message):
    if 'urgent' in message.content:
        message.priority = min(1.0, message.priority + 0.3)
    return message

urgent_rule = RoutingRule(
    message_type="*",
    priority_modifier=0.2,
    transform_function=priority_boost_transform
)

system.message_bus.add_routing_rule(urgent_rule)
```

### Attention Manipulation

```python
# Direct attention manipulation
from cognitive_grammar.atomspace.node import ConceptNode

# Create important concept
important_concept = ConceptNode("critical_knowledge")
important_concept.update_attention(0.9)  # High attention

# Add to hypergraph
concept_id = system.hypergraph.add_node(important_concept)

# Spread attention from this concept
system.hypergraph.spread_attention(concept_id, intensity=0.8)
```

## Configuration Options

### System Configuration

```python
config = SystemConfiguration()

# Agent configuration
config.memory_agents = 2          # Multiple memory agents
config.cognitive_agents = 3       # Multiple cognitive agents
config.enable_autonomy_agent = True

# Performance tuning
config.cycle_interval = 0.5       # Faster cycles
config.message_queue_size = 20000 # Larger queue
config.optimization_interval = 30 # More frequent optimization

# Backend selection
config.enable_scriptbots = True
config.enable_recurrentjs = True
config.enable_neuraltalk = False  # Disable if not needed
config.enable_transformers = True # Enable for large language models

# Hypergraph parameters
config.max_nodes = 50000
config.attention_decay_rate = 0.02

system = CognitiveGrammarSystem(config)
```

## Monitoring and Debugging

### Real-time System Monitoring

```python
async def monitor_system(system):
    while system.running:
        status = system.get_system_status()
        
        print(f"\n=== System Status ===")
        print(f"Uptime: {status['uptime']:.1f}s")
        print(f"Cycles: {status['cycles_completed']}")
        print(f"Messages: {status['total_messages_processed']}")
        print(f"Errors: {status['system_errors']}")
        
        # Agent details
        for agent_id, agent_status in status['agents'].items():
            print(f"  {agent_id}: {agent_status['state']} "
                  f"(tasks: {agent_status['performance']['tasks_completed']})")
        
        await asyncio.sleep(10)  # Update every 10 seconds

# Run monitoring alongside system
asyncio.create_task(monitor_system(system))
```

### Message Bus Debugging

```python
# Add message logging
def log_message_delivery(message, delivery_time):
    print(f"Delivered: {message.message_type} "
          f"from {message.sender_id} to {message.receiver_id} "
          f"in {delivery_time:.3f}s")

def log_message_error(message, error):
    print(f"Error: {message.message_type} failed - {error}")

system.message_bus.add_delivery_callback(log_message_delivery)
system.message_bus.add_error_callback(log_message_error)
```

### Hypergraph Visualization

```python
def visualize_attention_flow(hypergraph):
    """Simple attention flow visualization"""
    top_nodes = hypergraph.get_most_attended_nodes(10)
    
    print("\n=== High Attention Nodes ===")
    for node in top_nodes:
        print(f"{node.name}: {node.attention_value:.3f} "
              f"(type: {node.type.value})")
    
    print(f"\nTotal attention focus: {len(hypergraph.attention_focus)} nodes")

# Call periodically
visualize_attention_flow(system.hypergraph)
```

## Troubleshooting

### Common Issues

1. **System won't start**:
   - Check that all required agents are enabled in configuration
   - Verify Python dependencies are installed
   - Check log files for initialization errors

2. **Agents not communicating**:
   - Verify message bus is properly initialized
   - Check agent registration status
   - Monitor message queue for bottlenecks

3. **Poor performance**:
   - Increase cycle interval for stability
   - Reduce number of concurrent agents
   - Enable attention decay to clean up hypergraph

4. **Memory issues**:
   - Set lower max_nodes in hypergraph configuration
   - Enable periodic state saving
   - Monitor agent memory usage

### Debug Mode

```python
import logging

# Enable debug logging
config = SystemConfiguration()
config.log_level = logging.DEBUG
config.log_file = "debug.log"

system = CognitiveGrammarSystem(config)
```

This quick start guide provides the foundation for using the distributed cognitive grammar system. The system is designed to be extensible and can be customized for specific use cases while maintaining the core benefits of distributed intelligence and recursive self-modification.