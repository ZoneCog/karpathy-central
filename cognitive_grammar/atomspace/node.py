"""
Hypergraph Node Implementation
AtomSpace-inspired node system for distributed cognitive grammar
"""
import uuid
from typing import Any, Dict, List, Optional, Set
from enum import Enum
import time

class NodeType(Enum):
    """Types of nodes in the cognitive hypergraph"""
    CONCEPT = "concept"           # Abstract concepts
    PREDICATE = "predicate"       # Relationships/functions
    SCHEMA = "schema"             # Procedures/methods
    AGENT = "agent"               # Agent instances
    PATTERN = "pattern"           # Learned patterns
    TASK = "task"                 # Task definitions
    GOAL = "goal"                 # Goal states
    CONTEXT = "context"           # Contextual information
    MEMORY = "memory"             # Memory traces
    ATTENTION = "attention"       # Attention markers

class Node:
    """
    Basic node in the cognitive hypergraph
    Represents atomic units of knowledge, concepts, or entities
    """
    
    def __init__(self, 
                 name: str, 
                 node_type: NodeType,
                 value: Any = None,
                 confidence: float = 1.0,
                 attention_value: float = 0.0):
        self.id = str(uuid.uuid4())
        self.name = name
        self.type = node_type
        self.value = value
        self.confidence = confidence  # Truth value
        self.attention_value = attention_value  # ECAN attention
        self.created_at = time.time()
        self.modified_at = time.time()
        self.access_count = 0
        self.properties: Dict[str, Any] = {}
        self.incoming_links: Set[str] = set()  # Link IDs
        self.outgoing_links: Set[str] = set()  # Link IDs
        
    def add_property(self, key: str, value: Any):
        """Add a property to this node"""
        self.properties[key] = value
        self.modified_at = time.time()
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """Get a property value"""
        return self.properties.get(key, default)
    
    def access(self):
        """Record access to this node (for attention calculation)"""
        self.access_count += 1
        self.modified_at = time.time()
    
    def update_attention(self, delta: float):
        """Update attention value"""
        self.attention_value = max(0.0, min(1.0, self.attention_value + delta))
        self.modified_at = time.time()
    
    def add_incoming_link(self, link_id: str):
        """Add an incoming link"""
        self.incoming_links.add(link_id)
    
    def add_outgoing_link(self, link_id: str):
        """Add an outgoing link"""
        self.outgoing_links.add(link_id)
    
    def remove_incoming_link(self, link_id: str):
        """Remove an incoming link"""
        self.incoming_links.discard(link_id)
    
    def remove_outgoing_link(self, link_id: str):
        """Remove an outgoing link"""
        self.outgoing_links.discard(link_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type.value,
            'value': self.value,
            'confidence': self.confidence,
            'attention_value': self.attention_value,
            'created_at': self.created_at,
            'modified_at': self.modified_at,
            'access_count': self.access_count,
            'properties': self.properties,
            'incoming_links': list(self.incoming_links),
            'outgoing_links': list(self.outgoing_links)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        """Deserialize node from dictionary"""
        node = cls(
            name=data['name'],
            node_type=NodeType(data['type']),
            value=data.get('value'),
            confidence=data.get('confidence', 1.0),
            attention_value=data.get('attention_value', 0.0)
        )
        node.id = data['id']
        node.created_at = data.get('created_at', time.time())
        node.modified_at = data.get('modified_at', time.time())
        node.access_count = data.get('access_count', 0)
        node.properties = data.get('properties', {})
        node.incoming_links = set(data.get('incoming_links', []))
        node.outgoing_links = set(data.get('outgoing_links', []))
        return node
    
    def __str__(self) -> str:
        return f"Node({self.name}, {self.type.value}, attention={self.attention_value:.3f})"
    
    def __repr__(self) -> str:
        return self.__str__()

class ConceptNode(Node):
    """Specialized node for concepts"""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, NodeType.CONCEPT, **kwargs)

class PredicateNode(Node):
    """Specialized node for predicates/relationships"""
    
    def __init__(self, name: str, arity: int = 2, **kwargs):
        super().__init__(name, NodeType.PREDICATE, **kwargs)
        self.add_property('arity', arity)

class SchemaNode(Node):
    """Specialized node for schemas/procedures"""
    
    def __init__(self, name: str, procedure: Any = None, **kwargs):
        super().__init__(name, NodeType.SCHEMA, value=procedure, **kwargs)

class AgentNode(Node):
    """Specialized node for agent instances"""
    
    def __init__(self, name: str, agent_type: str, **kwargs):
        super().__init__(name, NodeType.AGENT, **kwargs)
        self.add_property('agent_type', agent_type)
        self.add_property('status', 'active')

class PatternNode(Node):
    """Specialized node for learned patterns"""
    
    def __init__(self, name: str, pattern_data: Any = None, **kwargs):
        super().__init__(name, NodeType.PATTERN, value=pattern_data, **kwargs)