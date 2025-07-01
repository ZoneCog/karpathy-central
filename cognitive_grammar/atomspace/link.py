"""
Hypergraph Link Implementation
Relationships and connections between nodes in the cognitive grammar
"""
import uuid
from typing import Any, Dict, List, Optional, Set
from enum import Enum
import time

class LinkType(Enum):
    """Types of links in the cognitive hypergraph"""
    INHERITANCE = "inheritance"     # IS-A relationships
    SIMILARITY = "similarity"       # Similarity/analogy
    EVALUATION = "evaluation"       # Predicate evaluations
    IMPLICATION = "implication"     # IF-THEN relationships
    SEQUENCE = "sequence"           # Temporal ordering
    EXECUTION = "execution"         # Procedure calls
    ATTENTION = "attention"         # Attention flow
    CONTEXT = "context"             # Contextual relationships
    MEMBERSHIP = "membership"       # Set membership
    COMPOSITION = "composition"     # Part-whole relationships
    CAUSATION = "causation"         # Causal relationships
    COMMUNICATION = "communication" # Agent communication

class Link:
    """
    Link connecting nodes in the cognitive hypergraph
    Represents relationships, dependencies, or interactions
    """
    
    def __init__(self,
                 link_type: LinkType,
                 source_nodes: List[str],  # Source node IDs
                 target_nodes: List[str],  # Target node IDs
                 strength: float = 1.0,
                 confidence: float = 1.0,
                 attention_value: float = 0.0):
        self.id = str(uuid.uuid4())
        self.type = link_type
        self.source_nodes = source_nodes
        self.target_nodes = target_nodes
        self.strength = strength  # Connection strength
        self.confidence = confidence  # Truth value
        self.attention_value = attention_value  # ECAN attention
        self.created_at = time.time()
        self.modified_at = time.time()
        self.activation_count = 0
        self.properties: Dict[str, Any] = {}
    
    def add_property(self, key: str, value: Any):
        """Add a property to this link"""
        self.properties[key] = value
        self.modified_at = time.time()
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """Get a property value"""
        return self.properties.get(key, default)
    
    def activate(self):
        """Record activation of this link"""
        self.activation_count += 1
        self.modified_at = time.time()
    
    def update_strength(self, delta: float):
        """Update connection strength"""
        self.strength = max(0.0, min(1.0, self.strength + delta))
        self.modified_at = time.time()
    
    def update_attention(self, delta: float):
        """Update attention value"""
        self.attention_value = max(0.0, min(1.0, self.attention_value + delta))
        self.modified_at = time.time()
    
    def get_all_nodes(self) -> List[str]:
        """Get all nodes involved in this link"""
        return self.source_nodes + self.target_nodes
    
    def involves_node(self, node_id: str) -> bool:
        """Check if this link involves a specific node"""
        return node_id in self.source_nodes or node_id in self.target_nodes
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize link to dictionary"""
        return {
            'id': self.id,
            'type': self.type.value,
            'source_nodes': self.source_nodes,
            'target_nodes': self.target_nodes,
            'strength': self.strength,
            'confidence': self.confidence,
            'attention_value': self.attention_value,
            'created_at': self.created_at,
            'modified_at': self.modified_at,
            'activation_count': self.activation_count,
            'properties': self.properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Link':
        """Deserialize link from dictionary"""
        link = cls(
            link_type=LinkType(data['type']),
            source_nodes=data['source_nodes'],
            target_nodes=data['target_nodes'],
            strength=data.get('strength', 1.0),
            confidence=data.get('confidence', 1.0),
            attention_value=data.get('attention_value', 0.0)
        )
        link.id = data['id']
        link.created_at = data.get('created_at', time.time())
        link.modified_at = data.get('modified_at', time.time())
        link.activation_count = data.get('activation_count', 0)
        link.properties = data.get('properties', {})
        return link
    
    def __str__(self) -> str:
        source_str = ','.join(self.source_nodes[:2])  # Show first 2 sources
        target_str = ','.join(self.target_nodes[:2])  # Show first 2 targets
        if len(self.source_nodes) > 2:
            source_str += '...'
        if len(self.target_nodes) > 2:
            target_str += '...'
        return f"Link({self.type.value}: [{source_str}] -> [{target_str}], strength={self.strength:.3f})"
    
    def __repr__(self) -> str:
        return self.__str__()

class InheritanceLink(Link):
    """Inheritance relationship (IS-A)"""
    
    def __init__(self, child_node: str, parent_node: str, **kwargs):
        super().__init__(LinkType.INHERITANCE, [child_node], [parent_node], **kwargs)

class SimilarityLink(Link):
    """Similarity/analogy relationship"""
    
    def __init__(self, node1: str, node2: str, similarity_score: float = 1.0, **kwargs):
        super().__init__(LinkType.SIMILARITY, [node1], [node2], strength=similarity_score, **kwargs)

class EvaluationLink(Link):
    """Predicate evaluation (predicate applied to arguments)"""
    
    def __init__(self, predicate_node: str, argument_nodes: List[str], truth_value: float = 1.0, **kwargs):
        super().__init__(LinkType.EVALUATION, [predicate_node], argument_nodes, 
                        strength=truth_value, **kwargs)

class ImplicationLink(Link):
    """Implication relationship (IF-THEN)"""
    
    def __init__(self, antecedent_nodes: List[str], consequent_nodes: List[str], **kwargs):
        super().__init__(LinkType.IMPLICATION, antecedent_nodes, consequent_nodes, **kwargs)

class SequenceLink(Link):
    """Temporal sequence relationship"""
    
    def __init__(self, before_nodes: List[str], after_nodes: List[str], **kwargs):
        super().__init__(LinkType.SEQUENCE, before_nodes, after_nodes, **kwargs)

class ExecutionLink(Link):
    """Procedure execution relationship"""
    
    def __init__(self, schema_node: str, input_nodes: List[str], output_nodes: List[str], **kwargs):
        super().__init__(LinkType.EXECUTION, [schema_node] + input_nodes, output_nodes, **kwargs)

class AttentionLink(Link):
    """Attention flow between cognitive elements"""
    
    def __init__(self, source_node: str, target_node: str, attention_flow: float = 1.0, **kwargs):
        super().__init__(LinkType.ATTENTION, [source_node], [target_node], 
                        strength=attention_flow, **kwargs)

class CommunicationLink(Link):
    """Communication between agents"""
    
    def __init__(self, sender_agent: str, receiver_agent: str, message_node: str, **kwargs):
        super().__init__(LinkType.COMMUNICATION, [sender_agent, message_node], [receiver_agent], **kwargs)