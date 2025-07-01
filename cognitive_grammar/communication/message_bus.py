"""
Distributed Message Bus Implementation
Inter-agent communication infrastructure for cognitive grammar system
"""
import asyncio
from typing import Any, Dict, List, Optional, Set, Callable
import time
import heapq
import json
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

from ..agents.base_agent import Message

@dataclass
class RoutingRule:
    """Message routing rule"""
    message_type: str
    source_pattern: Optional[str] = None
    target_pattern: Optional[str] = None
    priority_modifier: float = 0.0
    transform_function: Optional[Callable] = None

class MessageBus:
    """
    Distributed message bus for agent communication
    
    Features:
    - Priority-based message queuing
    - Message routing and filtering
    - Broadcast and multicast capabilities
    - Message persistence and replay
    - Load balancing and fault tolerance
    """
    
    def __init__(self, bus_id: str = "main_bus"):
        self.bus_id = bus_id
        
        # Message queuing
        self.message_queue = []  # Priority queue: (priority, timestamp, message)
        self.agent_mailboxes: Dict[str, deque] = defaultdict(deque)
        self.broadcast_queues: Dict[str, deque] = defaultdict(deque)  # topic -> messages
        
        # Agent registry
        self.registered_agents: Dict[str, Dict[str, Any]] = {}
        self.agent_capabilities: Dict[str, Set[str]] = {}
        self.agent_subscriptions: Dict[str, Set[str]] = defaultdict(set)  # agent -> topics
        
        # Routing
        self.routing_rules: List[RoutingRule] = []
        self.routing_table: Dict[str, str] = {}  # message_type -> default_agent
        self.load_balancing: Dict[str, List[str]] = defaultdict(list)  # capability -> agents
        
        # Message history and persistence
        self.message_history: deque = deque(maxlen=10000)
        self.failed_messages: deque = deque(maxlen=1000)
        self.message_stats: Dict[str, int] = defaultdict(int)
        
        # Configuration
        self.max_queue_size = 10000
        self.message_timeout = 30.0
        self.retry_attempts = 3
        self.batch_size = 100
        
        # Performance tracking
        self.messages_sent = 0
        self.messages_delivered = 0
        self.messages_failed = 0
        self.delivery_times = deque(maxlen=1000)
        
        # Event handlers
        self.message_handlers: Dict[str, Callable] = {}
        self.delivery_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
    
    def register_agent(self, 
                      agent_id: str, 
                      agent_type: str,
                      capabilities: List[str] = None,
                      subscriptions: List[str] = None) -> bool:
        """Register an agent with the message bus"""
        
        if agent_id in self.registered_agents:
            return False  # Already registered
        
        capabilities = capabilities or []
        subscriptions = subscriptions or []
        
        self.registered_agents[agent_id] = {
            'agent_type': agent_type,
            'capabilities': capabilities,
            'subscriptions': subscriptions,
            'registered_at': time.time(),
            'last_seen': time.time(),
            'messages_sent': 0,
            'messages_received': 0,
            'status': 'active'
        }
        
        self.agent_capabilities[agent_id] = set(capabilities)
        self.agent_subscriptions[agent_id] = set(subscriptions)
        
        # Update load balancing
        for capability in capabilities:
            self.load_balancing[capability].append(agent_id)
        
        # Create mailbox
        if agent_id not in self.agent_mailboxes:
            self.agent_mailboxes[agent_id] = deque()
        
        return True
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the message bus"""
        
        if agent_id not in self.registered_agents:
            return False
        
        agent_info = self.registered_agents[agent_id]
        
        # Remove from load balancing
        for capability in agent_info['capabilities']:
            if agent_id in self.load_balancing[capability]:
                self.load_balancing[capability].remove(agent_id)
        
        # Clean up
        del self.registered_agents[agent_id]
        del self.agent_capabilities[agent_id]
        del self.agent_subscriptions[agent_id]
        
        # Clear mailbox
        if agent_id in self.agent_mailboxes:
            del self.agent_mailboxes[agent_id]
        
        return True
    
    def send_message(self, message: Message) -> bool:
        """Send a message through the bus"""
        
        if len(self.message_queue) >= self.max_queue_size:
            self.messages_failed += 1
            return False
        
        # Apply routing rules
        processed_message = self._apply_routing_rules(message)
        
        # Add to priority queue
        priority_score = self._calculate_priority(processed_message)
        timestamp = time.time()
        
        heapq.heappush(
            self.message_queue,
            (-priority_score, timestamp, processed_message)  # Negative for max-heap
        )
        
        self.messages_sent += 1
        self.message_stats[message.message_type] += 1
        
        # Record in history
        self.message_history.append({
            'message': asdict(message),
            'timestamp': timestamp,
            'status': 'queued'
        })
        
        return True
    
    def broadcast(self, 
                  message_type: str,
                  content: Any,
                  sender_id: str = "system",
                  topic: str = "global",
                  priority: float = 0.5) -> int:
        """Broadcast message to all subscribers of a topic"""
        
        subscribers = self._get_topic_subscribers(topic)
        messages_sent = 0
        
        for subscriber_id in subscribers:
            message = Message(
                sender_id=sender_id,
                receiver_id=subscriber_id,
                message_type=message_type,
                content=content,
                priority=priority
            )
            
            if self.send_message(message):
                messages_sent += 1
        
        # Also add to broadcast queue for late subscribers
        broadcast_message = {
            'message_type': message_type,
            'content': content,
            'sender_id': sender_id,
            'timestamp': time.time(),
            'priority': priority
        }
        
        self.broadcast_queues[topic].append(broadcast_message)
        
        return messages_sent
    
    def multicast(self,
                  message_type: str,
                  content: Any,
                  target_agents: List[str],
                  sender_id: str = "system",
                  priority: float = 0.5) -> int:
        """Send message to multiple specific agents"""
        
        messages_sent = 0
        
        for target_id in target_agents:
            if target_id in self.registered_agents:
                message = Message(
                    sender_id=sender_id,
                    receiver_id=target_id,
                    message_type=message_type,
                    content=content,
                    priority=priority
                )
                
                if self.send_message(message):
                    messages_sent += 1
        
        return messages_sent
    
    def send_to_capability(self,
                          message_type: str,
                          content: Any,
                          required_capability: str,
                          sender_id: str = "system",
                          priority: float = 0.5,
                          load_balance: bool = True) -> bool:
        """Send message to agent(s) with specific capability"""
        
        capable_agents = self.load_balancing.get(required_capability, [])
        
        if not capable_agents:
            return False
        
        if load_balance:
            # Select agent with lowest current load
            target_agent = self._select_least_loaded_agent(capable_agents)
        else:
            # Send to all capable agents
            return self.multicast(message_type, content, capable_agents, sender_id, priority) > 0
        
        if target_agent:
            message = Message(
                sender_id=sender_id,
                receiver_id=target_agent,
                message_type=message_type,
                content=content,
                priority=priority
            )
            return self.send_message(message)
        
        return False
    
    async def deliver_messages(self) -> int:
        """Deliver queued messages to agents"""
        
        delivered_count = 0
        batch_count = 0
        
        while self.message_queue and batch_count < self.batch_size:
            priority, timestamp, message = heapq.heappop(self.message_queue)
            
            # Check message timeout
            if time.time() - timestamp > self.message_timeout:
                self._handle_message_timeout(message)
                continue
            
            # Deliver message
            if await self._deliver_message(message):
                delivered_count += 1
                self.messages_delivered += 1
                
                # Record delivery time
                delivery_time = time.time() - timestamp
                self.delivery_times.append(delivery_time)
                
                # Update agent stats
                if message.receiver_id in self.registered_agents:
                    self.registered_agents[message.receiver_id]['messages_received'] += 1
                
                # Call delivery callbacks
                for callback in self.delivery_callbacks:
                    try:
                        callback(message, delivery_time)
                    except Exception as e:
                        print(f"Delivery callback error: {e}")
            
            else:
                self.messages_failed += 1
                self.failed_messages.append({
                    'message': asdict(message),
                    'timestamp': timestamp,
                    'failure_reason': 'delivery_failed'
                })
            
            batch_count += 1
        
        return delivered_count
    
    async def _deliver_message(self, message: Message) -> bool:
        """Deliver a single message to its target"""
        
        receiver_id = message.receiver_id
        
        # Check if receiver is registered
        if receiver_id not in self.registered_agents:
            return False
        
        # Check if receiver is active
        agent_info = self.registered_agents[receiver_id]
        if agent_info['status'] != 'active':
            return False
        
        # Add to agent's mailbox
        self.agent_mailboxes[receiver_id].append(message)
        
        # Update last seen
        agent_info['last_seen'] = time.time()
        
        return True
    
    def get_messages(self, agent_id: str, max_messages: int = None) -> List[Message]:
        """Get messages for a specific agent"""
        
        if agent_id not in self.agent_mailboxes:
            return []
        
        mailbox = self.agent_mailboxes[agent_id]
        messages = []
        
        message_count = min(len(mailbox), max_messages) if max_messages else len(mailbox)
        
        for _ in range(message_count):
            if mailbox:
                messages.append(mailbox.popleft())
        
        return messages
    
    def peek_messages(self, agent_id: str, max_messages: int = 10) -> List[Message]:
        """Peek at messages without removing them from mailbox"""
        
        if agent_id not in self.agent_mailboxes:
            return []
        
        mailbox = self.agent_mailboxes[agent_id]
        return list(mailbox)[:max_messages]
    
    def subscribe_topic(self, agent_id: str, topic: str) -> bool:
        """Subscribe agent to a topic"""
        
        if agent_id not in self.registered_agents:
            return False
        
        self.agent_subscriptions[agent_id].add(topic)
        
        # Deliver any recent broadcast messages for this topic
        if topic in self.broadcast_queues:
            recent_messages = list(self.broadcast_queues[topic])[-10:]  # Last 10 messages
            
            for broadcast_msg in recent_messages:
                message = Message(
                    sender_id=broadcast_msg['sender_id'],
                    receiver_id=agent_id,
                    message_type=broadcast_msg['message_type'],
                    content=broadcast_msg['content'],
                    priority=broadcast_msg['priority']
                )
                self.send_message(message)
        
        return True
    
    def unsubscribe_topic(self, agent_id: str, topic: str) -> bool:
        """Unsubscribe agent from a topic"""
        
        if agent_id not in self.agent_subscriptions:
            return False
        
        self.agent_subscriptions[agent_id].discard(topic)
        return True
    
    def add_routing_rule(self, rule: RoutingRule):
        """Add a message routing rule"""
        self.routing_rules.append(rule)
    
    def set_routing_table(self, routing_table: Dict[str, str]):
        """Set default routing table"""
        self.routing_table.update(routing_table)
    
    def add_message_handler(self, message_type: str, handler: Callable):
        """Add a message handler for specific message type"""
        self.message_handlers[message_type] = handler
    
    def add_delivery_callback(self, callback: Callable):
        """Add callback for successful message delivery"""
        self.delivery_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable):
        """Add callback for message errors"""
        self.error_callbacks.append(callback)
    
    def _apply_routing_rules(self, message: Message) -> Message:
        """Apply routing rules to transform/route message"""
        
        for rule in self.routing_rules:
            if rule.message_type == message.message_type or rule.message_type == "*":
                # Check source pattern
                if rule.source_pattern and not self._matches_pattern(message.sender_id, rule.source_pattern):
                    continue
                
                # Check target pattern
                if rule.target_pattern and not self._matches_pattern(message.receiver_id, rule.target_pattern):
                    continue
                
                # Apply priority modifier
                message.priority += rule.priority_modifier
                message.priority = max(0.0, min(1.0, message.priority))
                
                # Apply transform function
                if rule.transform_function:
                    try:
                        message = rule.transform_function(message)
                    except Exception as e:
                        print(f"Routing rule transform error: {e}")
        
        return message
    
    def _matches_pattern(self, text: str, pattern: str) -> bool:
        """Check if text matches pattern (simple wildcard matching)"""
        if pattern == "*":
            return True
        if pattern.endswith("*"):
            return text.startswith(pattern[:-1])
        if pattern.startswith("*"):
            return text.endswith(pattern[1:])
        return text == pattern
    
    def _calculate_priority(self, message: Message) -> float:
        """Calculate message priority score"""
        base_priority = message.priority
        
        # Boost priority for certain message types
        priority_boosts = {
            'shutdown': 1.0,
            'emergency': 0.9,
            'health_check': 0.8,
            'task_assignment': 0.7,
            'status_request': 0.3
        }
        
        boost = priority_boosts.get(message.message_type, 0.0)
        return min(1.0, base_priority + boost)
    
    def _get_topic_subscribers(self, topic: str) -> List[str]:
        """Get list of agents subscribed to a topic"""
        subscribers = []
        
        for agent_id, subscriptions in self.agent_subscriptions.items():
            if topic in subscriptions or "global" in subscriptions:
                if self.registered_agents[agent_id]['status'] == 'active':
                    subscribers.append(agent_id)
        
        return subscribers
    
    def _select_least_loaded_agent(self, agent_ids: List[str]) -> Optional[str]:
        """Select agent with lowest current load"""
        
        active_agents = [
            agent_id for agent_id in agent_ids
            if (agent_id in self.registered_agents and 
                self.registered_agents[agent_id]['status'] == 'active')
        ]
        
        if not active_agents:
            return None
        
        # Calculate load based on mailbox size
        loads = {
            agent_id: len(self.agent_mailboxes.get(agent_id, []))
            for agent_id in active_agents
        }
        
        return min(loads, key=loads.get)
    
    def _handle_message_timeout(self, message: Message):
        """Handle message timeout"""
        self.failed_messages.append({
            'message': asdict(message),
            'timestamp': time.time(),
            'failure_reason': 'timeout'
        })
        
        # Call error callbacks
        for callback in self.error_callbacks:
            try:
                callback(message, 'timeout')
            except Exception as e:
                print(f"Error callback error: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get message bus statistics"""
        
        avg_delivery_time = (
            sum(self.delivery_times) / len(self.delivery_times)
            if self.delivery_times else 0.0
        )
        
        return {
            'bus_id': self.bus_id,
            'registered_agents': len(self.registered_agents),
            'active_agents': len([
                agent for agent in self.registered_agents.values()
                if agent['status'] == 'active'
            ]),
            'messages_sent': self.messages_sent,
            'messages_delivered': self.messages_delivered,
            'messages_failed': self.messages_failed,
            'queue_size': len(self.message_queue),
            'average_delivery_time': avg_delivery_time,
            'message_types': dict(self.message_stats),
            'failed_messages': len(self.failed_messages),
            'total_capabilities': len(self.load_balancing),
            'routing_rules': len(self.routing_rules)
        }
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific agent"""
        
        if agent_id not in self.registered_agents:
            return None
        
        agent_info = self.registered_agents[agent_id].copy()
        agent_info['mailbox_size'] = len(self.agent_mailboxes.get(agent_id, []))
        agent_info['subscriptions'] = list(self.agent_subscriptions.get(agent_id, set()))
        
        return agent_info
    
    def cleanup_inactive_agents(self, timeout: float = 300.0):
        """Remove agents that haven't been seen for a while"""
        
        current_time = time.time()
        inactive_agents = []
        
        for agent_id, agent_info in self.registered_agents.items():
            if current_time - agent_info['last_seen'] > timeout:
                inactive_agents.append(agent_id)
        
        for agent_id in inactive_agents:
            self.unregister_agent(agent_id)
        
        return len(inactive_agents)
    
    def save_state(self, filepath: str):
        """Save message bus state to file"""
        
        state = {
            'bus_id': self.bus_id,
            'registered_agents': self.registered_agents,
            'routing_table': self.routing_table,
            'message_stats': dict(self.message_stats),
            'statistics': self.get_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load message bus state from file"""
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.bus_id = state.get('bus_id', self.bus_id)
            
            # Restore agents
            for agent_id, agent_info in state.get('registered_agents', {}).items():
                self.register_agent(
                    agent_id,
                    agent_info['agent_type'],
                    agent_info.get('capabilities', []),
                    agent_info.get('subscriptions', [])
                )
            
            # Restore routing table
            self.routing_table.update(state.get('routing_table', {}))
            
            # Restore message stats
            for msg_type, count in state.get('message_stats', {}).items():
                self.message_stats[msg_type] = count
            
            return True
            
        except Exception as e:
            print(f"Error loading message bus state: {e}")
            return False