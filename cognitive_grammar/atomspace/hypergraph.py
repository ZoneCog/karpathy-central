"""
Hypergraph Implementation
Main hypergraph structure for distributed cognitive grammar
"""
from typing import Dict, List, Optional, Set, Any, Iterator, Tuple
import json
import time
from collections import defaultdict

from .node import Node, NodeType
from .link import Link, LinkType

class Hypergraph:
    """
    Distributed hypergraph for cognitive knowledge representation
    AtomSpace-inspired structure supporting nodes and hyperlinks
    """
    
    def __init__(self, name: str = "CognitiveHypergraph"):
        self.name = name
        self.nodes: Dict[str, Node] = {}  # node_id -> Node
        self.links: Dict[str, Link] = {}  # link_id -> Link
        self.node_index: Dict[str, Set[str]] = defaultdict(set)  # name -> node_ids
        self.type_index: Dict[NodeType, Set[str]] = defaultdict(set)  # type -> node_ids
        self.link_type_index: Dict[LinkType, Set[str]] = defaultdict(set)  # type -> link_ids
        self.created_at = time.time()
        self.modified_at = time.time()
        
        # Attention tracking
        self.attention_focus: Set[str] = set()  # Currently attended nodes
        self.attention_spread_rate = 0.1
        
    def add_node(self, node: Node) -> str:
        """Add a node to the hypergraph"""
        if node.id in self.nodes:
            return node.id  # Already exists
            
        self.nodes[node.id] = node
        self.node_index[node.name].add(node.id)
        self.type_index[node.type].add(node.id)
        self.modified_at = time.time()
        return node.id
    
    def add_link(self, link: Link) -> str:
        """Add a link to the hypergraph"""
        if link.id in self.links:
            return link.id  # Already exists
            
        # Verify all referenced nodes exist
        all_nodes = link.get_all_nodes()
        for node_id in all_nodes:
            if node_id not in self.nodes:
                raise ValueError(f"Node {node_id} not found in hypergraph")
        
        self.links[link.id] = link
        self.link_type_index[link.type].add(link.id)
        
        # Update node link references
        for node_id in link.source_nodes:
            self.nodes[node_id].add_outgoing_link(link.id)
        for node_id in link.target_nodes:
            self.nodes[node_id].add_incoming_link(link.id)
            
        self.modified_at = time.time()
        return link.id
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID"""
        node = self.nodes.get(node_id)
        if node:
            node.access()
        return node
    
    def get_link(self, link_id: str) -> Optional[Link]:
        """Get a link by ID"""
        link = self.links.get(link_id)
        if link:
            link.activate()
        return link
    
    def find_nodes_by_name(self, name: str) -> List[Node]:
        """Find nodes by name"""
        node_ids = self.node_index.get(name, set())
        return [self.nodes[nid] for nid in node_ids]
    
    def find_nodes_by_type(self, node_type: NodeType) -> List[Node]:
        """Find nodes by type"""
        node_ids = self.type_index.get(node_type, set())
        return [self.nodes[nid] for nid in node_ids]
    
    def find_links_by_type(self, link_type: LinkType) -> List[Link]:
        """Find links by type"""
        link_ids = self.link_type_index.get(link_type, set())
        return [self.links[lid] for lid in link_ids]
    
    def get_incoming_links(self, node_id: str) -> List[Link]:
        """Get all incoming links to a node"""
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self.links[lid] for lid in node.incoming_links if lid in self.links]
    
    def get_outgoing_links(self, node_id: str) -> List[Link]:
        """Get all outgoing links from a node"""
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self.links[lid] for lid in node.outgoing_links if lid in self.links]
    
    def get_neighbors(self, node_id: str) -> Set[str]:
        """Get all neighboring nodes (connected by any link)"""
        neighbors = set()
        
        # Outgoing links
        for link in self.get_outgoing_links(node_id):
            neighbors.update(link.target_nodes)
            
        # Incoming links
        for link in self.get_incoming_links(node_id):
            neighbors.update(link.source_nodes)
            
        neighbors.discard(node_id)  # Remove self
        return neighbors
    
    def find_path(self, start_node_id: str, end_node_id: str, max_depth: int = 5) -> Optional[List[str]]:
        """Find a path between two nodes using BFS"""
        if start_node_id == end_node_id:
            return [start_node_id]
            
        visited = {start_node_id}
        queue = [(start_node_id, [start_node_id])]
        
        for _ in range(max_depth):
            if not queue:
                break
                
            current_id, path = queue.pop(0)
            
            for neighbor_id in self.get_neighbors(current_id):
                if neighbor_id == end_node_id:
                    return path + [neighbor_id]
                    
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))
                    
        return None
    
    def spread_attention(self, source_node_id: str, intensity: float = 1.0):
        """Spread attention from a source node to its neighbors"""
        if source_node_id not in self.nodes:
            return
            
        source_node = self.nodes[source_node_id]
        source_node.update_attention(intensity)
        self.attention_focus.add(source_node_id)
        
        # Spread to immediate neighbors
        for neighbor_id in self.get_neighbors(source_node_id):
            neighbor = self.nodes[neighbor_id]
            spread_amount = intensity * self.attention_spread_rate
            neighbor.update_attention(spread_amount)
            
            if spread_amount > 0.1:  # Threshold for attention focus
                self.attention_focus.add(neighbor_id)
    
    def decay_attention(self, decay_rate: float = 0.01):
        """Decay attention values over time"""
        to_remove = set()
        
        for node_id in list(self.attention_focus):
            node = self.nodes[node_id]
            node.update_attention(-decay_rate)
            
            if node.attention_value < 0.01:
                to_remove.add(node_id)
                
        self.attention_focus -= to_remove
    
    def get_most_attended_nodes(self, count: int = 10) -> List[Node]:
        """Get the most attended nodes"""
        all_nodes = list(self.nodes.values())
        return sorted(all_nodes, key=lambda n: n.attention_value, reverse=True)[:count]
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its links"""
        if node_id not in self.nodes:
            return False
            
        node = self.nodes[node_id]
        
        # Remove all links involving this node
        links_to_remove = list(node.incoming_links) + list(node.outgoing_links)
        for link_id in links_to_remove:
            self.remove_link(link_id)
        
        # Remove from indices
        self.node_index[node.name].discard(node_id)
        self.type_index[node.type].discard(node_id)
        
        # Remove the node
        del self.nodes[node_id]
        self.attention_focus.discard(node_id)
        self.modified_at = time.time()
        return True
    
    def remove_link(self, link_id: str) -> bool:
        """Remove a link from the hypergraph"""
        if link_id not in self.links:
            return False
            
        link = self.links[link_id]
        
        # Remove from node references
        for node_id in link.source_nodes:
            if node_id in self.nodes:
                self.nodes[node_id].remove_outgoing_link(link_id)
        for node_id in link.target_nodes:
            if node_id in self.nodes:
                self.nodes[node_id].remove_incoming_link(link_id)
        
        # Remove from indices
        self.link_type_index[link.type].discard(link_id)
        
        # Remove the link
        del self.links[link_id]
        self.modified_at = time.time()
        return True
    
    def query(self, 
              node_types: Optional[List[NodeType]] = None,
              link_types: Optional[List[LinkType]] = None,
              min_attention: float = 0.0,
              properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the hypergraph for nodes and links matching criteria"""
        
        matching_nodes = []
        matching_links = []
        
        # Filter nodes
        for node in self.nodes.values():
            if node_types and node.type not in node_types:
                continue
            if node.attention_value < min_attention:
                continue
            if properties:
                if not all(node.get_property(k) == v for k, v in properties.items()):
                    continue
            matching_nodes.append(node)
        
        # Filter links
        for link in self.links.values():
            if link_types and link.type not in link_types:
                continue
            if link.attention_value < min_attention:
                continue
            matching_links.append(link)
        
        return {
            'nodes': matching_nodes,
            'links': matching_links,
            'query_time': time.time()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get hypergraph statistics"""
        node_type_counts = {}
        for node_type, node_ids in self.type_index.items():
            node_type_counts[node_type.value] = len(node_ids)
            
        link_type_counts = {}
        for link_type, link_ids in self.link_type_index.items():
            link_type_counts[link_type.value] = len(link_ids)
        
        return {
            'name': self.name,
            'total_nodes': len(self.nodes),
            'total_links': len(self.links),
            'node_types': node_type_counts,
            'link_types': link_type_counts,
            'attention_focus_size': len(self.attention_focus),
            'created_at': self.created_at,
            'modified_at': self.modified_at
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize hypergraph to dictionary"""
        return {
            'name': self.name,
            'nodes': {nid: node.to_dict() for nid, node in self.nodes.items()},
            'links': {lid: link.to_dict() for lid, link in self.links.items()},
            'created_at': self.created_at,
            'modified_at': self.modified_at,
            'attention_focus': list(self.attention_focus)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Hypergraph':
        """Deserialize hypergraph from dictionary"""
        hg = cls(data.get('name', 'CognitiveHypergraph'))
        hg.created_at = data.get('created_at', time.time())
        hg.modified_at = data.get('modified_at', time.time())
        
        # Load nodes first
        for node_data in data.get('nodes', {}).values():
            node = Node.from_dict(node_data)
            hg.nodes[node.id] = node
            hg.node_index[node.name].add(node.id)
            hg.type_index[node.type].add(node.id)
        
        # Then load links
        for link_data in data.get('links', {}).values():
            link = Link.from_dict(link_data)
            hg.links[link.id] = link
            hg.link_type_index[link.type].add(link.id)
        
        hg.attention_focus = set(data.get('attention_focus', []))
        return hg
    
    def __len__(self) -> int:
        return len(self.nodes) + len(self.links)
    
    def __str__(self) -> str:
        return f"Hypergraph({self.name}: {len(self.nodes)} nodes, {len(self.links)} links)"