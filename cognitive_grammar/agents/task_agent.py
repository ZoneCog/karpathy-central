"""
Task Manager Agent Implementation
Orchestrates distributed task allocation and execution
"""
import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple
import time
import uuid
from enum import Enum
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentState, Message
from ..atomspace.hypergraph import Hypergraph
from ..atomspace.node import Node, NodeType
from ..atomspace.link import Link, LinkType

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Task:
    """Task representation"""
    id: str
    name: str
    task_type: str
    requirements: List[str]  # Required capabilities
    input_data: Dict[str, Any]
    priority: TaskPriority
    deadline: Optional[float] = None
    parent_task_id: Optional[str] = None
    subtask_ids: List[str] = None
    assigned_agent_id: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.subtask_ids is None:
            self.subtask_ids = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'task_type': self.task_type,
            'requirements': self.requirements,
            'input_data': self.input_data,
            'priority': self.priority.value,
            'deadline': self.deadline,
            'parent_task_id': self.parent_task_id,
            'subtask_ids': self.subtask_ids,
            'assigned_agent_id': self.assigned_agent_id,
            'status': self.status.value,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'result': self.result,
            'error': self.error
        }

class TaskManagerAgent(BaseAgent):
    """
    Task Manager Agent - Recursive Task Orchestration
    
    Responsibilities:
    - Decompose complex goals into executable tasks
    - Assign tasks to appropriate agents based on capabilities
    - Monitor task progress and handle failures
    - Optimize resource allocation and scheduling
    - Manage task dependencies and coordination
    """
    
    def __init__(self, agent_id: str, hypergraph: Hypergraph):
        capabilities = [
            'task_decomposition',
            'agent_assignment',
            'progress_monitoring',
            'resource_allocation',
            'dependency_management',
            'failure_recovery',
            'load_balancing'
        ]
        super().__init__(agent_id, "TaskManagerAgent", hypergraph, capabilities)
        
        # Task management
        self.tasks: Dict[str, Task] = {}
        self.task_queue: List[str] = []  # Task IDs in priority order
        self.active_tasks: Set[str] = set()
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        
        # Agent registry
        self.available_agents: Dict[str, Dict[str, Any]] = {}
        self.agent_workload: Dict[str, int] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        
        # Performance tracking
        self.tasks_dispatched = 0
        self.tasks_decomposed = 0
        self.average_completion_time = 0.0
        self.resource_utilization = 0.0
        
        # Scheduling parameters
        self.max_concurrent_tasks = 10
        self.decomposition_depth_limit = 5
        self.task_timeout = 300.0  # 5 minutes default
    
    def _initialize(self):
        """Initialize task manager components"""
        # Register task-specific message handlers
        self.message_handlers.update({
            'submit_task': self._handle_submit_task,
            'decompose_task': self._handle_decompose_task,
            'agent_registration': self._handle_agent_registration,
            'task_progress': self._handle_task_progress,
            'task_completion': self._handle_task_completion,
            'task_failure': self._handle_task_failure,
            'agent_status_update': self._handle_agent_status_update,
            'cancel_task': self._handle_cancel_task,
            'get_task_status': self._handle_get_task_status
        })
        
        # Discover existing agents in hypergraph
        self._discover_agents()
        
        self.state = AgentState.ACTIVE
    
    def _discover_agents(self):
        """Discover available agents in the hypergraph"""
        agent_nodes = self.hypergraph.find_nodes_by_type(NodeType.AGENT)
        
        for agent_node in agent_nodes:
            if agent_node.id != self.agent_node.id:  # Don't include self
                agent_type = agent_node.get_property('agent_type', 'unknown')
                capabilities = agent_node.get_property('capabilities', [])
                status = agent_node.get_property('status', 'unknown')
                
                self.available_agents[agent_node.id] = {
                    'agent_type': agent_type,
                    'capabilities': capabilities,
                    'status': status,
                    'last_seen': time.time()
                }
                self.agent_capabilities[agent_node.id] = capabilities
                self.agent_workload[agent_node.id] = 0
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process task manager operations"""
        task_type = task.get('type')
        start_time = time.time()
        
        try:
            if task_type == 'submit_goal':
                result = await self._submit_goal(task)
            elif task_type == 'decompose_goal':
                result = await self._decompose_goal(task)
            elif task_type == 'schedule_tasks':
                result = await self._schedule_tasks(task)
            elif task_type == 'monitor_progress':
                result = await self._monitor_progress(task)
            elif task_type == 'rebalance_load':
                result = await self._rebalance_load(task)
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
    
    async def _submit_goal(self, goal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a high-level goal for decomposition and execution"""
        goal_name = goal_data.get('name')
        goal_description = goal_data.get('description')
        requirements = goal_data.get('requirements', [])
        input_data = goal_data.get('input_data', {})
        priority = TaskPriority(goal_data.get('priority', TaskPriority.NORMAL.value))
        deadline = goal_data.get('deadline')
        
        # Create root task
        root_task = Task(
            id=str(uuid.uuid4()),
            name=goal_name,
            task_type='goal',
            requirements=requirements,
            input_data=input_data,
            priority=priority,
            deadline=deadline
        )
        
        self.tasks[root_task.id] = root_task
        
        # Decompose the goal into subtasks
        await self._decompose_task_recursive(root_task)
        
        # Schedule the tasks
        await self._schedule_pending_tasks()
        
        return {
            'goal_id': root_task.id,
            'subtasks_created': len(root_task.subtask_ids),
            'status': 'submitted'
        }
    
    async def _decompose_goal(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose a goal into executable subtasks"""
        task_id = task_data.get('task_id')
        task = self.tasks.get(task_id)
        
        if not task:
            return {'error': f'Task {task_id} not found'}
        
        subtasks = await self._decompose_task_recursive(task)
        
        return {
            'task_id': task_id,
            'subtasks_created': len(subtasks),
            'decomposition_depth': self._get_task_depth(task)
        }
    
    async def _decompose_task_recursive(self, task: Task, depth: int = 0) -> List[Task]:
        """Recursively decompose a task into subtasks"""
        if depth >= self.decomposition_depth_limit:
            return []
        
        subtasks = []
        
        # Task decomposition logic based on task type
        if task.task_type == 'goal':
            subtasks = await self._decompose_goal_task(task)
        elif task.task_type == 'neural_processing':
            subtasks = await self._decompose_neural_task(task)
        elif task.task_type == 'memory_operation':
            subtasks = await self._decompose_memory_task(task)
        elif task.task_type == 'communication':
            subtasks = await self._decompose_communication_task(task)
        else:
            # For unknown task types, try to find appropriate agent
            return []
        
        # Create and register subtasks
        for subtask_data in subtasks:
            subtask = Task(
                id=str(uuid.uuid4()),
                name=subtask_data['name'],
                task_type=subtask_data['task_type'],
                requirements=subtask_data.get('requirements', []),
                input_data=subtask_data.get('input_data', {}),
                priority=task.priority,
                deadline=task.deadline,
                parent_task_id=task.id
            )
            
            self.tasks[subtask.id] = subtask
            task.subtask_ids.append(subtask.id)
            
            # Recursively decompose if needed
            if subtask_data.get('decompose', False):
                await self._decompose_task_recursive(subtask, depth + 1)
        
        self.tasks_decomposed += 1
        return [self.tasks[sid] for sid in task.subtask_ids]
    
    async def _decompose_goal_task(self, task: Task) -> List[Dict[str, Any]]:
        """Decompose a high-level goal into subtasks"""
        goal_type = task.input_data.get('goal_type', 'general')
        
        if goal_type == 'learning':
            return [
                {
                    'name': f'prepare_data_{task.name}',
                    'task_type': 'memory_operation',
                    'requirements': ['knowledge_storage', 'context_retrieval'],
                    'input_data': {'operation': 'prepare_learning_data', 'source': task.input_data}
                },
                {
                    'name': f'neural_training_{task.name}',
                    'task_type': 'neural_processing',
                    'requirements': ['learning', 'neural_networks'],
                    'input_data': {'operation': 'train_model', 'config': task.input_data}
                },
                {
                    'name': f'validate_results_{task.name}',
                    'task_type': 'neural_processing',
                    'requirements': ['evaluation', 'pattern_recognition'],
                    'input_data': {'operation': 'validate_model', 'criteria': task.input_data}
                }
            ]
        elif goal_type == 'reasoning':
            return [
                {
                    'name': f'gather_context_{task.name}',
                    'task_type': 'memory_operation',
                    'requirements': ['context_retrieval', 'knowledge_storage'],
                    'input_data': {'operation': 'retrieve_context', 'query': task.input_data}
                },
                {
                    'name': f'inference_{task.name}',
                    'task_type': 'neural_processing',
                    'requirements': ['reasoning', 'inference'],
                    'input_data': {'operation': 'inference', 'context': task.input_data}
                }
            ]
        else:
            # Generic decomposition
            return [
                {
                    'name': f'analyze_{task.name}',
                    'task_type': 'neural_processing',
                    'requirements': ['pattern_recognition'],
                    'input_data': task.input_data
                },
                {
                    'name': f'execute_{task.name}',
                    'task_type': 'neural_processing',
                    'requirements': ['execution'],
                    'input_data': task.input_data
                }
            ]
    
    async def _decompose_neural_task(self, task: Task) -> List[Dict[str, Any]]:
        """Decompose neural processing tasks"""
        operation = task.input_data.get('operation', 'process')
        
        if operation == 'train_model':
            return [
                {
                    'name': f'prepare_network_{task.name}',
                    'task_type': 'neural_setup',
                    'requirements': ['neural_networks'],
                    'input_data': {'operation': 'setup_network', 'config': task.input_data}
                },
                {
                    'name': f'training_loop_{task.name}',
                    'task_type': 'neural_training',
                    'requirements': ['learning', 'optimization'],
                    'input_data': {'operation': 'training_loop', 'config': task.input_data}
                }
            ]
        else:
            return []  # Atomic neural task
    
    async def _decompose_memory_task(self, task: Task) -> List[Dict[str, Any]]:
        """Decompose memory operation tasks"""
        # Memory tasks are typically atomic
        return []
    
    async def _decompose_communication_task(self, task: Task) -> List[Dict[str, Any]]:
        """Decompose communication tasks"""
        # Communication tasks are typically atomic
        return []
    
    def _get_task_depth(self, task: Task) -> int:
        """Get the decomposition depth of a task"""
        if not task.subtask_ids:
            return 0
        
        max_subtask_depth = 0
        for subtask_id in task.subtask_ids:
            subtask = self.tasks.get(subtask_id)
            if subtask:
                subtask_depth = self._get_task_depth(subtask)
                max_subtask_depth = max(max_subtask_depth, subtask_depth)
        
        return 1 + max_subtask_depth
    
    async def _schedule_pending_tasks(self):
        """Schedule pending tasks to available agents"""
        # Get all pending tasks
        pending_tasks = [
            task for task in self.tasks.values()
            if task.status == TaskStatus.PENDING and not task.subtask_ids
        ]
        
        # Sort by priority and deadline
        pending_tasks.sort(key=lambda t: (
            -t.priority.value,
            t.deadline if t.deadline else float('inf'),
            t.created_at
        ))
        
        for task in pending_tasks:
            if len(self.active_tasks) >= self.max_concurrent_tasks:
                break
            
            # Find suitable agent
            agent_id = self._find_suitable_agent(task)
            if agent_id:
                await self._assign_task(task, agent_id)
    
    def _find_suitable_agent(self, task: Task) -> Optional[str]:
        """Find the most suitable agent for a task"""
        suitable_agents = []
        
        for agent_id, capabilities in self.agent_capabilities.items():
            # Check if agent has required capabilities
            if all(req in capabilities for req in task.requirements):
                workload = self.agent_workload.get(agent_id, 0)
                agent_info = self.available_agents.get(agent_id, {})
                
                if agent_info.get('status') == 'active':
                    suitable_agents.append((agent_id, workload))
        
        if not suitable_agents:
            return None
        
        # Select agent with lowest workload
        suitable_agents.sort(key=lambda x: x[1])
        return suitable_agents[0][0]
    
    async def _assign_task(self, task: Task, agent_id: str):
        """Assign a task to an agent"""
        task.assigned_agent_id = agent_id
        task.status = TaskStatus.ASSIGNED
        task.started_at = time.time()
        
        self.active_tasks.add(task.id)
        self.agent_workload[agent_id] = self.agent_workload.get(agent_id, 0) + 1
        self.tasks_dispatched += 1
        
        # Send task to agent
        self.send_message(
            receiver_id=agent_id,
            message_type='task_assignment',
            content=task.to_dict(),
            priority=task.priority.value / 4.0  # Convert to 0-1 range
        )
    
    async def _schedule_tasks(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule tasks based on current workload"""
        await self._schedule_pending_tasks()
        
        return {
            'tasks_scheduled': len(self.active_tasks),
            'pending_tasks': len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
            'agent_workloads': dict(self.agent_workload)
        }
    
    async def _monitor_progress(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor progress of active tasks"""
        current_time = time.time()
        overdue_tasks = []
        
        for task_id in list(self.active_tasks):
            task = self.tasks[task_id]
            
            # Check for timeouts
            if (task.deadline and current_time > task.deadline) or \
               (task.started_at and current_time - task.started_at > self.task_timeout):
                overdue_tasks.append(task_id)
                task.status = TaskStatus.FAILED
                task.error = "Task timeout"
                task.completed_at = current_time
                
                self.active_tasks.remove(task_id)
                self.failed_tasks.add(task_id)
                
                # Free up agent workload
                if task.assigned_agent_id:
                    self.agent_workload[task.assigned_agent_id] -= 1
        
        return {
            'active_tasks': len(self.active_tasks),
            'overdue_tasks': len(overdue_tasks),
            'failed_tasks': len(self.failed_tasks),
            'completed_tasks': len(self.completed_tasks)
        }
    
    async def _rebalance_load(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Rebalance workload across agents"""
        # Calculate average workload
        total_workload = sum(self.agent_workload.values())
        num_agents = len(self.available_agents)
        
        if num_agents == 0:
            return {'message': 'No agents available for rebalancing'}
        
        avg_workload = total_workload / num_agents
        
        # Find overloaded and underloaded agents
        overloaded = [(aid, load) for aid, load in self.agent_workload.items() if load > avg_workload * 1.5]
        underloaded = [(aid, load) for aid, load in self.agent_workload.items() if load < avg_workload * 0.5]
        
        # TODO: Implement task migration logic
        # For now, just reschedule pending tasks
        await self._schedule_pending_tasks()
        
        return {
            'average_workload': avg_workload,
            'overloaded_agents': len(overloaded),
            'underloaded_agents': len(underloaded),
            'rebalancing_performed': True
        }
    
    # Message handlers
    async def _handle_submit_task(self, message: Message) -> Optional[Message]:
        """Handle task submission"""
        result = await self._submit_goal(message.content)
        
        return Message(
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            message_type='task_submitted',
            content=result
        )
    
    async def _handle_decompose_task(self, message: Message) -> Optional[Message]:
        """Handle task decomposition requests"""
        result = await self._decompose_goal(message.content)
        
        return Message(
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            message_type='task_decomposed',
            content=result
        )
    
    async def _handle_agent_registration(self, message: Message) -> Optional[Message]:
        """Handle agent registration"""
        agent_info = message.content
        agent_id = agent_info.get('agent_id')
        
        self.available_agents[agent_id] = agent_info
        self.agent_capabilities[agent_id] = agent_info.get('capabilities', [])
        self.agent_workload[agent_id] = 0
        
        return Message(
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            message_type='registration_confirmed',
            content={'status': 'registered'}
        )
    
    async def _handle_task_completion(self, message: Message) -> Optional[Message]:
        """Handle task completion notifications"""
        task_id = message.content.get('task_id')
        result = message.content.get('result')
        
        task = self.tasks.get(task_id)
        if task:
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = time.time()
            
            self.active_tasks.discard(task_id)
            self.completed_tasks.add(task_id)
            
            # Free up agent workload
            if task.assigned_agent_id:
                self.agent_workload[task.assigned_agent_id] -= 1
            
            # Check if parent task is complete
            if task.parent_task_id:
                await self._check_parent_completion(task.parent_task_id)
        
        return None
    
    async def _handle_task_failure(self, message: Message) -> Optional[Message]:
        """Handle task failure notifications"""
        task_id = message.content.get('task_id')
        error = message.content.get('error')
        
        task = self.tasks.get(task_id)
        if task:
            task.status = TaskStatus.FAILED
            task.error = error
            task.completed_at = time.time()
            
            self.active_tasks.discard(task_id)
            self.failed_tasks.add(task_id)
            
            # Free up agent workload
            if task.assigned_agent_id:
                self.agent_workload[task.assigned_agent_id] -= 1
            
            # TODO: Implement failure recovery strategies
        
        return None
    
    async def _check_parent_completion(self, parent_task_id: str):
        """Check if a parent task is complete based on its subtasks"""
        parent_task = self.tasks.get(parent_task_id)
        if not parent_task:
            return
        
        all_complete = True
        for subtask_id in parent_task.subtask_ids:
            subtask = self.tasks.get(subtask_id)
            if subtask and subtask.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                all_complete = False
                break
        
        if all_complete:
            parent_task.status = TaskStatus.COMPLETED
            parent_task.completed_at = time.time()
            
            # Recursively check grandparent
            if parent_task.parent_task_id:
                await self._check_parent_completion(parent_task.parent_task_id)
    
    async def _handle_agent_status_update(self, message: Message) -> Optional[Message]:
        """Handle agent status updates"""
        agent_id = message.sender_id
        status_info = message.content
        
        if agent_id in self.available_agents:
            self.available_agents[agent_id].update(status_info)
            self.available_agents[agent_id]['last_seen'] = time.time()
        
        return None
    
    async def _handle_cancel_task(self, message: Message) -> Optional[Message]:
        """Handle task cancellation requests"""
        task_id = message.content.get('task_id')
        
        task = self.tasks.get(task_id)
        if task and task.status in [TaskStatus.PENDING, TaskStatus.ASSIGNED]:
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            
            self.active_tasks.discard(task_id)
            
            # Notify assigned agent if any
            if task.assigned_agent_id:
                self.send_message(
                    receiver_id=task.assigned_agent_id,
                    message_type='task_cancelled',
                    content={'task_id': task_id}
                )
                self.agent_workload[task.assigned_agent_id] -= 1
            
            result = {'status': 'cancelled'}
        else:
            result = {'error': 'Task not found or cannot be cancelled'}
        
        return Message(
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            message_type='task_cancellation_response',
            content=result
        )
    
    async def _handle_get_task_status(self, message: Message) -> Optional[Message]:
        """Handle task status requests"""
        task_id = message.content.get('task_id')
        
        task = self.tasks.get(task_id)
        if task:
            result = task.to_dict()
        else:
            result = {'error': 'Task not found'}
        
        return Message(
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            message_type='task_status_response',
            content=result
        )
    
    async def _process_cycle(self):
        """Task manager processing cycle"""
        # Periodic maintenance tasks
        
        # 1. Schedule pending tasks
        await self._schedule_pending_tasks()
        
        # 2. Monitor task progress
        await self._monitor_progress({})
        
        # 3. Update resource utilization
        if self.available_agents:
            total_capacity = len(self.available_agents) * self.max_concurrent_tasks
            current_load = len(self.active_tasks)
            self.resource_utilization = current_load / total_capacity if total_capacity > 0 else 0
        
        # 4. Update average completion time
        completed_tasks_with_time = [
            t for t in self.tasks.values()
            if t.status == TaskStatus.COMPLETED and t.started_at and t.completed_at
        ]
        
        if completed_tasks_with_time:
            total_time = sum(t.completed_at - t.started_at for t in completed_tasks_with_time)
            self.average_completion_time = total_time / len(completed_tasks_with_time)
    
    def get_status(self) -> Dict[str, Any]:
        """Get task manager status"""
        base_status = self.get_base_status()
        
        task_status = {
            'total_tasks': len(self.tasks),
            'active_tasks': len(self.active_tasks),
            'pending_tasks': len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'tasks_dispatched': self.tasks_dispatched,
            'tasks_decomposed': self.tasks_decomposed,
            'average_completion_time': self.average_completion_time,
            'resource_utilization': self.resource_utilization,
            'available_agents': len(self.available_agents),
            'agent_workloads': dict(self.agent_workload)
        }
        
        base_status.update(task_status)
        return base_status