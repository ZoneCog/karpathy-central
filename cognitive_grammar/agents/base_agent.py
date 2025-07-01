"""
Base Agent Interface
Foundation for distributed cognitive grammar agents
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
import uuid
import time
import asyncio
from enum import Enum

from ..atomspace.hypergraph import Hypergraph
from ..atomspace.node import Node, NodeType, AgentNode
from ..atomspace.link import Link, LinkType, CommunicationLink

class AgentState(Enum):
    """Agent execution states"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PROCESSING = "processing"
    WAITING = "waiting"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class Message:
    """Inter-agent communication message"""
    
    def __init__(self, 
                 sender_id: str,
                 receiver_id: str,
                 message_type: str,
                 content: Any,
                 priority: float = 0.5):
        self.id = str(uuid.uuid4())
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.message_type = message_type
        self.content = content
        self.priority = priority
        self.timestamp = time.time()
        self.processed = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id,
            'message_type': self.message_type,
            'content': self.content,
            'priority': self.priority,
            'timestamp': self.timestamp,
            'processed': self.processed
        }

class BaseAgent(ABC):
    """
    Base class for all cognitive grammar agents
    Provides common functionality for distributed agent network
    """
    
    def __init__(self, 
                 agent_id: str,
                 agent_type: str,
                 hypergraph: Hypergraph,
                 capabilities: List[str] = None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.hypergraph = hypergraph
        self.capabilities = capabilities or []
        self.state = AgentState.INITIALIZING
        self.created_at = time.time()
        self.last_activity = time.time()
        
        # Message handling
        self.inbox: List[Message] = []
        self.outbox: List[Message] = []
        self.message_handlers: Dict[str, callable] = {}
        
        # Performance metrics
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_processing_time = 0.0
        self.attention_allocation = 0.0
        
        # Agent properties
        self.properties: Dict[str, Any] = {}
        
        # Create agent node in hypergraph
        self.agent_node = AgentNode(
            name=f"{agent_type}_{agent_id}",
            agent_type=agent_type,
            confidence=1.0
        )
        self.agent_node.add_property('capabilities', capabilities)
        self.agent_node.add_property('created_at', self.created_at)
        self.hypergraph.add_node(self.agent_node)
        
        # Register default message handlers
        self._register_default_handlers()
        
        # Initialize agent-specific components
        self._initialize()
    
    @abstractmethod
    def _initialize(self):
        """Initialize agent-specific components"""
        pass
    
    @abstractmethod
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a cognitive task"""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        pass
    
    def _register_default_handlers(self):
        """Register default message handlers"""
        self.message_handlers.update({
            'ping': self._handle_ping,
            'status_request': self._handle_status_request,
            'shutdown': self._handle_shutdown,
            'attention_update': self._handle_attention_update
        })
    
    async def _handle_ping(self, message: Message) -> Optional[Message]:
        """Handle ping messages"""
        return Message(
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            message_type='pong',
            content={'timestamp': time.time()}
        )
    
    async def _handle_status_request(self, message: Message) -> Optional[Message]:
        """Handle status request messages"""
        return Message(
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            message_type='status_response',
            content=self.get_status()
        )
    
    async def _handle_shutdown(self, message: Message) -> Optional[Message]:
        """Handle shutdown messages"""
        self.state = AgentState.SHUTDOWN
        return None
    
    async def _handle_attention_update(self, message: Message) -> Optional[Message]:
        """Handle attention allocation updates"""
        new_allocation = message.content.get('attention_allocation', 0.0)
        self.attention_allocation = max(0.0, min(1.0, new_allocation))
        self.agent_node.update_attention(self.attention_allocation)
        return None
    
    def send_message(self, 
                     receiver_id: str, 
                     message_type: str, 
                     content: Any, 
                     priority: float = 0.5):
        """Send a message to another agent"""
        message = Message(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            priority=priority
        )
        self.outbox.append(message)
        
        # Create communication link in hypergraph
        receiver_nodes = self.hypergraph.find_nodes_by_name(f"*_{receiver_id}")
        if receiver_nodes:
            # Create message node
            message_node = Node(
                name=f"message_{message.id}",
                node_type=NodeType.CONTEXT,
                value=message.to_dict()
            )
            message_node_id = self.hypergraph.add_node(message_node)
            
            # Create communication link
            comm_link = CommunicationLink(
                sender_agent=self.agent_node.id,
                receiver_agent=receiver_nodes[0].id,
                message_node=message_node_id
            )
            self.hypergraph.add_link(comm_link)
    
    def receive_message(self, message: Message):
        """Receive a message from another agent"""
        self.inbox.append(message)
    
    async def process_messages(self):
        """Process all messages in inbox"""
        while self.inbox and self.state != AgentState.SHUTDOWN:
            # Sort by priority (highest first)
            self.inbox.sort(key=lambda m: m.priority, reverse=True)
            message = self.inbox.pop(0)
            
            handler = self.message_handlers.get(message.message_type)
            if handler:
                try:
                    response = await handler(message)
                    if response:
                        self.outbox.append(response)
                except Exception as e:
                    print(f"Error processing message {message.id}: {e}")
            
            message.processed = True
            self.last_activity = time.time()
    
    def update_attention(self, attention_value: float):
        """Update agent's attention allocation"""
        self.attention_allocation = max(0.0, min(1.0, attention_value))
        self.agent_node.update_attention(self.attention_allocation)
        
        # Spread attention in hypergraph
        if attention_value > 0.5:
            self.hypergraph.spread_attention(self.agent_node.id, attention_value)
    
    def add_capability(self, capability: str):
        """Add a new capability to the agent"""
        if capability not in self.capabilities:
            self.capabilities.append(capability)
            self.agent_node.add_property('capabilities', self.capabilities)
    
    def has_capability(self, capability: str) -> bool:
        """Check if agent has a specific capability"""
        return capability in self.capabilities
    
    def record_task_completion(self, success: bool, processing_time: float):
        """Record task completion metrics"""
        if success:
            self.tasks_completed += 1
        else:
            self.tasks_failed += 1
        self.total_processing_time += processing_time
        self.last_activity = time.time()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        total_tasks = self.tasks_completed + self.tasks_failed
        success_rate = self.tasks_completed / total_tasks if total_tasks > 0 else 0.0
        avg_processing_time = self.total_processing_time / total_tasks if total_tasks > 0 else 0.0
        
        return {
            'tasks_completed': self.tasks_completed,
            'tasks_failed': self.tasks_failed,
            'success_rate': success_rate,
            'average_processing_time': avg_processing_time,
            'total_processing_time': self.total_processing_time,
            'attention_allocation': self.attention_allocation
        }
    
    def set_property(self, key: str, value: Any):
        """Set an agent property"""
        self.properties[key] = value
        self.agent_node.add_property(key, value)
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """Get an agent property"""
        return self.properties.get(key, default)
    
    async def run_cycle(self):
        """Run one agent execution cycle"""
        if self.state == AgentState.SHUTDOWN:
            return
            
        self.state = AgentState.ACTIVE
        
        try:
            # Process incoming messages
            await self.process_messages()
            
            # Perform agent-specific processing
            await self._process_cycle()
            
            # Update state
            self.state = AgentState.WAITING
            
        except Exception as e:
            print(f"Error in agent {self.agent_id} cycle: {e}")
            self.state = AgentState.ERROR
    
    @abstractmethod
    async def _process_cycle(self):
        """Agent-specific processing cycle"""
        pass
    
    def get_base_status(self) -> Dict[str, Any]:
        """Get base agent status information"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'state': self.state.value,
            'capabilities': self.capabilities,
            'created_at': self.created_at,
            'last_activity': self.last_activity,
            'inbox_size': len(self.inbox),
            'outbox_size': len(self.outbox),
            'attention_allocation': self.attention_allocation,
            'performance': self.get_performance_metrics()
        }
    
    def shutdown(self):
        """Shutdown the agent"""
        self.state = AgentState.SHUTDOWN
        self.agent_node.add_property('shutdown_at', time.time())
    
    def __str__(self) -> str:
        return f"{self.agent_type}({self.agent_id}, {self.state.value})"