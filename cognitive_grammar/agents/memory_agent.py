"""
Memory Agent Implementation
Manages distributed knowledge representation in the hypergraph
"""
import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple
import time
import json

from .base_agent import BaseAgent, AgentState, Message
from ..atomspace.hypergraph import Hypergraph
from ..atomspace.node import Node, NodeType, ConceptNode, PatternNode
from ..atomspace.link import Link, LinkType, InheritanceLink, SimilarityLink

class MemoryAgent(BaseAgent):
    """
    Memory Agent - Distributed Knowledge Representation
    
    Responsibilities:
    - Store and retrieve knowledge patterns
    - Maintain hypergraph structure integrity
    - Provide contextual information for tasks
    - Manage knowledge evolution and pruning
    """
    
    def __init__(self, agent_id: str, hypergraph: Hypergraph):
        capabilities = [
            'knowledge_storage',
            'pattern_recognition',
            'context_retrieval',
            'knowledge_pruning',
            'similarity_search',
            'concept_formation'
        ]
        super().__init__(agent_id, "MemoryAgent", hypergraph, capabilities)
        
        # Memory-specific properties
        self.knowledge_domains: Set[str] = set()
        self.pattern_cache: Dict[str, Any] = {}
        self.concept_hierarchy: Dict[str, Set[str]] = {}  # concept -> subconcepts
        self.similarity_threshold = 0.7
        self.pruning_threshold = 0.1  # Remove low-attention nodes
        
        # Performance tracking
        self.retrieval_count = 0
        self.storage_count = 0
        self.pruning_count = 0
    
    def _initialize(self):
        """Initialize memory agent components"""
        # Register memory-specific message handlers
        self.message_handlers.update({
            'store_knowledge': self._handle_store_knowledge,
            'retrieve_knowledge': self._handle_retrieve_knowledge,
            'find_similar': self._handle_find_similar,
            'get_context': self._handle_get_context,
            'prune_knowledge': self._handle_prune_knowledge,
            'form_concept': self._handle_form_concept
        })
        
        # Initialize concept hierarchy
        self._build_concept_hierarchy()
        
        self.state = AgentState.ACTIVE
    
    def _build_concept_hierarchy(self):
        """Build concept hierarchy from existing inheritance links"""
        inheritance_links = self.hypergraph.find_links_by_type(LinkType.INHERITANCE)
        
        for link in inheritance_links:
            if len(link.source_nodes) == 1 and len(link.target_nodes) == 1:
                child_node = self.hypergraph.get_node(link.source_nodes[0])
                parent_node = self.hypergraph.get_node(link.target_nodes[0])
                
                if child_node and parent_node:
                    if parent_node.name not in self.concept_hierarchy:
                        self.concept_hierarchy[parent_node.name] = set()
                    self.concept_hierarchy[parent_node.name].add(child_node.name)
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process memory-related tasks"""
        task_type = task.get('type')
        start_time = time.time()
        
        try:
            if task_type == 'store_pattern':
                result = await self._store_pattern(task)
            elif task_type == 'retrieve_context':
                result = await self._retrieve_context(task)
            elif task_type == 'find_associations':
                result = await self._find_associations(task)
            elif task_type == 'update_knowledge':
                result = await self._update_knowledge(task)
            elif task_type == 'prune_memory':
                result = await self._prune_memory(task)
            else:
                result = {'error': f'Unknown task type: {task_type}'}
            
            processing_time = time.time() - start_time
            self.record_task_completion(True, processing_time)
            
            return {
                'status': 'completed',
                'result': result,
                'processing_time': processing_time
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.record_task_completion(False, processing_time)
            
            return {
                'status': 'failed',
                'error': str(e),
                'processing_time': processing_time
            }
    
    async def _store_pattern(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Store a learned pattern in the hypergraph"""
        pattern_data = task.get('pattern_data')
        pattern_type = task.get('pattern_type', 'general')
        confidence = task.get('confidence', 1.0)
        domain = task.get('domain', 'general')
        
        # Create pattern node
        pattern_name = f"pattern_{pattern_type}_{len(self.pattern_cache)}"
        pattern_node = PatternNode(
            name=pattern_name,
            pattern_data=pattern_data,
            confidence=confidence
        )
        pattern_node.add_property('domain', domain)
        pattern_node.add_property('pattern_type', pattern_type)
        
        pattern_id = self.hypergraph.add_node(pattern_node)
        self.pattern_cache[pattern_id] = pattern_data
        self.knowledge_domains.add(domain)
        self.storage_count += 1
        
        # Create links to related concepts if specified
        related_concepts = task.get('related_concepts', [])
        for concept_name in related_concepts:
            concept_nodes = self.hypergraph.find_nodes_by_name(concept_name)
            if not concept_nodes:
                # Create new concept
                concept_node = ConceptNode(concept_name)
                concept_id = self.hypergraph.add_node(concept_node)
            else:
                concept_id = concept_nodes[0].id
            
            # Create similarity link
            similarity_link = SimilarityLink(
                pattern_id,
                concept_id,
                similarity_score=confidence
            )
            self.hypergraph.add_link(similarity_link)
        
        return {
            'pattern_id': pattern_id,
            'pattern_name': pattern_name,
            'stored_at': time.time()
        }
    
    async def _retrieve_context(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve contextual knowledge for a query"""
        query = task.get('query')
        context_size = task.get('context_size', 10)
        domain_filter = task.get('domain')
        
        relevant_nodes = []
        
        # Find nodes matching query terms
        if isinstance(query, str):
            query_terms = query.lower().split()
            for node in self.hypergraph.nodes.values():
                node_text = node.name.lower()
                if any(term in node_text for term in query_terms):
                    relevant_nodes.append(node)
        
        # Filter by domain if specified
        if domain_filter:
            relevant_nodes = [
                node for node in relevant_nodes
                if node.get_property('domain') == domain_filter
            ]
        
        # Sort by attention value and confidence
        relevant_nodes.sort(
            key=lambda n: (n.attention_value, n.confidence),
            reverse=True
        )
        
        # Get top nodes and their neighbors
        context_nodes = relevant_nodes[:context_size]
        context_links = []
        
        for node in context_nodes:
            # Add incoming and outgoing links
            context_links.extend(self.hypergraph.get_incoming_links(node.id))
            context_links.extend(self.hypergraph.get_outgoing_links(node.id))
        
        self.retrieval_count += 1
        
        return {
            'context_nodes': [node.to_dict() for node in context_nodes],
            'context_links': [link.to_dict() for link in context_links],
            'total_relevant': len(relevant_nodes),
            'retrieved_at': time.time()
        }
    
    async def _find_associations(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Find associations between concepts"""
        source_concept = task.get('source_concept')
        max_distance = task.get('max_distance', 3)
        min_strength = task.get('min_strength', 0.1)
        
        source_nodes = self.hypergraph.find_nodes_by_name(source_concept)
        if not source_nodes:
            return {'associations': [], 'message': 'Source concept not found'}
        
        source_node = source_nodes[0]
        associations = []
        
        # BFS to find associated concepts
        visited = {source_node.id}
        queue = [(source_node.id, 0)]  # (node_id, distance)
        
        while queue:
            current_id, distance = queue.pop(0)
            
            if distance >= max_distance:
                continue
            
            # Check all links from current node
            for link in self.hypergraph.get_outgoing_links(current_id):
                if link.strength < min_strength:
                    continue
                
                for target_id in link.target_nodes:
                    if target_id not in visited:
                        visited.add(target_id)
                        target_node = self.hypergraph.get_node(target_id)
                        
                        if target_node:
                            associations.append({
                                'node': target_node.to_dict(),
                                'distance': distance + 1,
                                'strength': link.strength,
                                'link_type': link.type.value
                            })
                            
                            queue.append((target_id, distance + 1))
        
        # Sort by strength and distance
        associations.sort(key=lambda a: (a['strength'], -a['distance']), reverse=True)
        
        return {
            'source_concept': source_concept,
            'associations': associations,
            'total_found': len(associations)
        }
    
    async def _update_knowledge(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Update existing knowledge with new information"""
        node_id = task.get('node_id')
        updates = task.get('updates', {})
        
        node = self.hypergraph.get_node(node_id)
        if not node:
            return {'error': f'Node {node_id} not found'}
        
        # Update node properties
        for key, value in updates.items():
            if key == 'confidence':
                node.confidence = max(0.0, min(1.0, value))
            elif key == 'attention':
                node.update_attention(value)
            else:
                node.add_property(key, value)
        
        return {
            'node_id': node_id,
            'updated_properties': list(updates.keys()),
            'updated_at': time.time()
        }
    
    async def _prune_memory(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Prune low-attention knowledge from memory"""
        threshold = task.get('threshold', self.pruning_threshold)
        dry_run = task.get('dry_run', False)
        
        nodes_to_prune = []
        links_to_prune = []
        
        # Find low-attention nodes
        for node in self.hypergraph.nodes.values():
            if (node.attention_value < threshold and 
                node.access_count < 5 and  # Haven't been accessed much
                time.time() - node.created_at > 3600):  # At least 1 hour old
                nodes_to_prune.append(node)
        
        # Find links with low strength
        for link in self.hypergraph.links.values():
            if (link.strength < threshold and
                link.activation_count < 3):
                links_to_prune.append(link)
        
        if not dry_run:
            # Actually remove nodes and links
            for node in nodes_to_prune:
                self.hypergraph.remove_node(node.id)
            for link in links_to_prune:
                self.hypergraph.remove_link(link.id)
            
            self.pruning_count += len(nodes_to_prune) + len(links_to_prune)
        
        return {
            'nodes_pruned': len(nodes_to_prune),
            'links_pruned': len(links_to_prune),
            'dry_run': dry_run,
            'threshold_used': threshold
        }
    
    async def _handle_store_knowledge(self, message: Message) -> Optional[Message]:
        """Handle knowledge storage requests"""
        result = await self._store_pattern(message.content)
        
        return Message(
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            message_type='knowledge_stored',
            content=result
        )
    
    async def _handle_retrieve_knowledge(self, message: Message) -> Optional[Message]:
        """Handle knowledge retrieval requests"""
        result = await self._retrieve_context(message.content)
        
        return Message(
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            message_type='knowledge_retrieved',
            content=result
        )
    
    async def _handle_find_similar(self, message: Message) -> Optional[Message]:
        """Handle similarity search requests"""
        result = await self._find_associations(message.content)
        
        return Message(
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            message_type='similarities_found',
            content=result
        )
    
    async def _handle_get_context(self, message: Message) -> Optional[Message]:
        """Handle context requests"""
        result = await self._retrieve_context(message.content)
        
        return Message(
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            message_type='context_provided',
            content=result
        )
    
    async def _handle_prune_knowledge(self, message: Message) -> Optional[Message]:
        """Handle knowledge pruning requests"""
        result = await self._prune_memory(message.content)
        
        return Message(
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            message_type='knowledge_pruned',
            content=result
        )
    
    async def _handle_form_concept(self, message: Message) -> Optional[Message]:
        """Handle concept formation requests"""
        concept_name = message.content.get('concept_name')
        instances = message.content.get('instances', [])
        
        # Create concept node
        concept_node = ConceptNode(concept_name)
        concept_id = self.hypergraph.add_node(concept_node)
        
        # Create inheritance links from instances
        for instance_name in instances:
            instance_nodes = self.hypergraph.find_nodes_by_name(instance_name)
            if instance_nodes:
                inheritance_link = InheritanceLink(
                    instance_nodes[0].id,
                    concept_id
                )
                self.hypergraph.add_link(inheritance_link)
        
        result = {
            'concept_id': concept_id,
            'concept_name': concept_name,
            'instances_linked': len(instances)
        }
        
        return Message(
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            message_type='concept_formed',
            content=result
        )
    
    async def _process_cycle(self):
        """Memory agent processing cycle"""
        # Periodic maintenance tasks
        
        # 1. Update attention decay
        self.hypergraph.decay_attention(decay_rate=0.01)
        
        # 2. Check for automatic pruning (every 100 cycles)
        if self.tasks_completed % 100 == 0:
            await self._prune_memory({'threshold': self.pruning_threshold})
        
        # 3. Update concept hierarchy
        if self.tasks_completed % 50 == 0:
            self._build_concept_hierarchy()
        
        # 4. Cache most accessed patterns
        most_attended = self.hypergraph.get_most_attended_nodes(10)
        for node in most_attended:
            if node.type == NodeType.PATTERN:
                self.pattern_cache[node.id] = node.value
    
    def get_status(self) -> Dict[str, Any]:
        """Get memory agent status"""
        base_status = self.get_base_status()
        
        memory_status = {
            'knowledge_domains': list(self.knowledge_domains),
            'concept_hierarchy_size': len(self.concept_hierarchy),
            'pattern_cache_size': len(self.pattern_cache),
            'retrieval_count': self.retrieval_count,
            'storage_count': self.storage_count,
            'pruning_count': self.pruning_count,
            'hypergraph_stats': self.hypergraph.get_statistics()
        }
        
        base_status.update(memory_status)
        return base_status