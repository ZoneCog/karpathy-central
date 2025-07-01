"""
Cognitive Grammar - Distributed Agentic Cognitive Grammar System

A distributed network of intelligent agents coordinating through hypergraph-based
knowledge representation, enabling emergent intelligence and recursive self-modification.

Core Components:
- Hypergraph knowledge representation (AtomSpace-inspired)
- Multi-agent cognitive processing framework
- Distributed message passing and coordination
- Attention-based resource allocation
- Integration with existing neural architectures

Key Features:
- Recursive task decomposition and allocation
- Self-modifying attention mechanisms
- Cross-system knowledge transfer
- Dynamic agent orchestration
- Real-time system optimization
"""

from .orchestrator import CognitiveGrammarSystem, SystemConfiguration
from .atomspace.hypergraph import Hypergraph
from .atomspace.node import Node, NodeType
from .atomspace.link import Link, LinkType

from .agents.base_agent import BaseAgent, Message
from .agents.memory_agent import MemoryAgent
from .agents.task_agent import TaskManagerAgent
from .agents.cognitive_agent import CognitiveAgent
from .agents.autonomy_agent import AutonomyAgent

from .communication.message_bus import MessageBus

__version__ = "1.0.0"
__author__ = "ZoneCog Team"
__description__ = "Distributed Agentic Cognitive Grammar System"

__all__ = [
    # Core system
    'CognitiveGrammarSystem',
    'SystemConfiguration',
    
    # Hypergraph components
    'Hypergraph',
    'Node',
    'NodeType', 
    'Link',
    'LinkType',
    
    # Agent framework
    'BaseAgent',
    'Message',
    'MemoryAgent',
    'TaskManagerAgent', 
    'CognitiveAgent',
    'AutonomyAgent',
    
    # Communication
    'MessageBus',
]

def create_minimal_system():
    """Create a minimal cognitive grammar system for quick testing"""
    config = SystemConfiguration()
    config.memory_agents = 1
    config.cognitive_agents = 1
    return CognitiveGrammarSystem(config)

def create_full_system():
    """Create a full-featured cognitive grammar system"""
    config = SystemConfiguration()
    config.enable_memory_agent = True
    config.enable_task_manager = True  
    config.enable_cognitive_agent = True
    config.enable_autonomy_agent = True
    config.memory_agents = 2
    config.cognitive_agents = 2
    return CognitiveGrammarSystem(config)