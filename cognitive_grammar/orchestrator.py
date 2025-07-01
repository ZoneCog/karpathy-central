"""
Cognitive Grammar System Orchestrator
Main orchestration layer for the distributed agentic cognitive grammar system
"""
import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple, Type
import time
import logging
import signal
import sys
import os

from .atomspace.hypergraph import Hypergraph
from .agents.base_agent import BaseAgent, AgentState
from .agents.memory_agent import MemoryAgent
from .agents.task_agent import TaskManagerAgent
from .agents.cognitive_agent import CognitiveAgent
from .agents.autonomy_agent import AutonomyAgent
from .communication.message_bus import MessageBus, RoutingRule

class SystemConfiguration:
    """System configuration settings"""
    
    def __init__(self):
        # Agent configuration
        self.enable_memory_agent = True
        self.enable_task_manager = True
        self.enable_cognitive_agent = True
        self.enable_autonomy_agent = True
        
        # Multiple agent instances
        self.memory_agents = 1
        self.cognitive_agents = 1
        
        # Hypergraph configuration
        self.initial_attention_budget = 1000.0
        self.attention_decay_rate = 0.01
        self.max_nodes = 10000
        self.max_links = 50000
        
        # Message bus configuration
        self.message_queue_size = 10000
        self.message_timeout = 30.0
        self.batch_delivery_size = 100
        
        # System parameters
        self.cycle_interval = 1.0  # seconds between main cycles
        self.health_check_interval = 10.0
        self.optimization_interval = 60.0
        self.save_state_interval = 300.0  # 5 minutes
        
        # Logging
        self.log_level = logging.INFO
        self.log_file = "cognitive_system.log"
        
        # Integration settings
        self.enable_scriptbots = True
        self.enable_recurrentjs = True
        self.enable_neuraltalk = True
        self.enable_transformers = False  # Requires additional setup

class CognitiveGrammarSystem:
    """
    Main orchestrator for the distributed cognitive grammar system
    
    Coordinates all agents, manages system lifecycle, and provides
    unified interface to the distributed intelligence network.
    """
    
    def __init__(self, config: SystemConfiguration = None):
        self.config = config or SystemConfiguration()
        
        # Core components
        self.hypergraph = None
        self.message_bus = None
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_types: Dict[str, Type[BaseAgent]] = {}
        
        # System state
        self.running = False
        self.startup_time = None
        self.last_health_check = 0
        self.last_optimization = 0
        self.last_state_save = 0
        
        # Performance tracking
        self.cycles_completed = 0
        self.total_messages_processed = 0
        self.system_errors = 0
        
        # Event loop
        self.event_loop = None
        self.background_tasks = []
        
        # Setup logging
        self._setup_logging()
        
        # Register signal handlers
        self._setup_signal_handlers()
    
    def _setup_logging(self):
        """Setup system logging"""
        logging.basicConfig(
            level=self.config.log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('CognitiveGrammarSystem')
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def initialize(self) -> bool:
        """Initialize the cognitive grammar system"""
        
        try:
            self.logger.info("Initializing Cognitive Grammar System...")
            
            # Initialize core components
            await self._initialize_hypergraph()
            await self._initialize_message_bus()
            await self._initialize_agents()
            await self._setup_agent_communication()
            
            # Validate system integrity
            if not await self._validate_system():
                raise Exception("System validation failed")
            
            self.logger.info("System initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            return False
    
    async def _initialize_hypergraph(self):
        """Initialize the knowledge hypergraph"""
        
        self.hypergraph = Hypergraph("CognitiveGrammarKnowledge")
        self.hypergraph.attention_spread_rate = self.config.attention_decay_rate
        
        self.logger.info("Hypergraph initialized")
    
    async def _initialize_message_bus(self):
        """Initialize the message bus"""
        
        self.message_bus = MessageBus("cognitive_bus")
        self.message_bus.max_queue_size = self.config.message_queue_size
        self.message_bus.message_timeout = self.config.message_timeout
        self.message_bus.batch_size = self.config.batch_delivery_size
        
        # Setup routing rules
        self._setup_routing_rules()
        
        self.logger.info("Message bus initialized")
    
    def _setup_routing_rules(self):
        """Setup message routing rules"""
        
        # Priority rules for critical messages
        critical_rule = RoutingRule(
            message_type="emergency",
            priority_modifier=0.5
        )
        self.message_bus.add_routing_rule(critical_rule)
        
        # Health check routing
        health_rule = RoutingRule(
            message_type="health_check",
            target_pattern="*AutonomyAgent*",
            priority_modifier=0.3
        )
        self.message_bus.add_routing_rule(health_rule)
        
        # Task routing to task manager
        task_rule = RoutingRule(
            message_type="submit_task",
            target_pattern="*TaskManagerAgent*",
            priority_modifier=0.2
        )
        self.message_bus.add_routing_rule(task_rule)
    
    async def _initialize_agents(self):
        """Initialize all system agents"""
        
        agent_id_counter = 0
        
        # Memory Agent(s)
        if self.config.enable_memory_agent:
            for i in range(self.config.memory_agents):
                agent_id = f"memory_agent_{agent_id_counter:03d}"
                agent = MemoryAgent(agent_id, self.hypergraph)
                await self._register_agent(agent)
                agent_id_counter += 1
        
        # Task Manager Agent
        if self.config.enable_task_manager:
            agent_id = f"task_manager_{agent_id_counter:03d}"
            agent = TaskManagerAgent(agent_id, self.hypergraph)
            await self._register_agent(agent)
            agent_id_counter += 1
        
        # Cognitive Agent(s)
        if self.config.enable_cognitive_agent:
            backends = []
            if self.config.enable_scriptbots:
                backends.append('scriptbots')
            if self.config.enable_recurrentjs:
                backends.append('recurrentjs')
            if self.config.enable_neuraltalk:
                backends.append('neuraltalk')
            if self.config.enable_transformers:
                backends.append('transformers')
            
            for i in range(self.config.cognitive_agents):
                agent_id = f"cognitive_agent_{agent_id_counter:03d}"
                agent = CognitiveAgent(agent_id, self.hypergraph, backends)
                await self._register_agent(agent)
                agent_id_counter += 1
        
        # Autonomy Agent
        if self.config.enable_autonomy_agent:
            agent_id = f"autonomy_agent_{agent_id_counter:03d}"
            agent = AutonomyAgent(agent_id, self.hypergraph)
            await self._register_agent(agent)
            agent_id_counter += 1
        
        self.logger.info(f"Initialized {len(self.agents)} agents")
    
    async def _register_agent(self, agent: BaseAgent):
        """Register an agent with the system"""
        
        # Add to agent registry
        self.agents[agent.agent_id] = agent
        
        # Register with message bus
        self.message_bus.register_agent(
            agent.agent_id,
            agent.agent_type,
            agent.capabilities,
            ["global"]  # Subscribe to global topic
        )
        
        # Register agent type
        self.agent_types[agent.agent_type] = type(agent)
        
        self.logger.info(f"Registered agent: {agent.agent_id} ({agent.agent_type})")
    
    async def _setup_agent_communication(self):
        """Setup communication between agents"""
        
        # Agent discovery - let agents know about each other
        agent_registry = {}
        for agent_id, agent in self.agents.items():
            agent_registry[agent_id] = {
                'agent_type': agent.agent_type,
                'capabilities': agent.capabilities
            }
        
        # Broadcast agent registry to all agents
        self.message_bus.broadcast(
            message_type="agent_registry_update",
            content={'agent_registry': agent_registry},
            topic="global"
        )
        
        # Setup default routing
        routing_table = {}
        
        # Route to first available agent of each type
        for agent_id, agent in self.agents.items():
            if agent.agent_type not in routing_table:
                routing_table[agent.agent_type] = agent_id
        
        self.message_bus.set_routing_table(routing_table)
        
        self.logger.info("Agent communication setup completed")
    
    async def _validate_system(self) -> bool:
        """Validate system integrity"""
        
        # Check essential agents
        essential_types = ['MemoryAgent', 'TaskManagerAgent', 'AutonomyAgent']
        
        for agent_type in essential_types:
            if not any(agent.agent_type == agent_type for agent in self.agents.values()):
                if (agent_type == 'MemoryAgent' and self.config.enable_memory_agent) or \
                   (agent_type == 'TaskManagerAgent' and self.config.enable_task_manager) or \
                   (agent_type == 'AutonomyAgent' and self.config.enable_autonomy_agent):
                    self.logger.error(f"Essential agent type {agent_type} not found")
                    return False
        
        # Check hypergraph
        if not self.hypergraph:
            self.logger.error("Hypergraph not initialized")
            return False
        
        # Check message bus
        if not self.message_bus:
            self.logger.error("Message bus not initialized")
            return False
        
        # Test agent communication
        test_successful = await self._test_agent_communication()
        if not test_successful:
            self.logger.error("Agent communication test failed")
            return False
        
        return True
    
    async def _test_agent_communication(self) -> bool:
        """Test basic agent communication"""
        
        try:
            # Send ping to all agents
            ping_count = self.message_bus.broadcast(
                message_type="ping",
                content={'test': True},
                topic="global"
            )
            
            if ping_count == 0:
                return False
            
            # Process messages briefly
            await self.message_bus.deliver_messages()
            
            # Check if agents received messages
            for agent_id in self.agents.keys():
                messages = self.message_bus.get_messages(agent_id, 1)
                if messages:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Communication test error: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the cognitive grammar system"""
        
        if self.running:
            self.logger.warning("System already running")
            return False
        
        # Initialize if not already done
        if not self.hypergraph or not self.message_bus:
            if not await self.initialize():
                return False
        
        self.running = True
        self.startup_time = time.time()
        
        # Start background tasks
        await self._start_background_tasks()
        
        # Start main system loop
        self.logger.info("Starting main system loop")
        await self._main_loop()
        
        return True
    
    async def _start_background_tasks(self):
        """Start background system tasks"""
        
        # Message delivery task
        delivery_task = asyncio.create_task(self._message_delivery_loop())
        self.background_tasks.append(delivery_task)
        
        # System monitoring task
        monitoring_task = asyncio.create_task(self._system_monitoring_loop())
        self.background_tasks.append(monitoring_task)
        
        # State persistence task
        persistence_task = asyncio.create_task(self._state_persistence_loop())
        self.background_tasks.append(persistence_task)
        
        self.logger.info(f"Started {len(self.background_tasks)} background tasks")
    
    async def _main_loop(self):
        """Main system execution loop"""
        
        while self.running:
            try:
                cycle_start = time.time()
                
                # Run agent cycles
                await self._run_agent_cycles()
                
                # Process any immediate tasks
                await self._process_immediate_tasks()
                
                # Update system metrics
                self._update_system_metrics()
                
                # Calculate cycle time and wait
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, self.config.cycle_interval - cycle_time)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
                self.cycles_completed += 1
                
                # Periodic logging
                if self.cycles_completed % 100 == 0:
                    self.logger.info(f"Completed {self.cycles_completed} cycles")
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                self.system_errors += 1
                await asyncio.sleep(1)  # Brief pause on error
    
    async def _run_agent_cycles(self):
        """Run processing cycles for all agents"""
        
        # Create tasks for all agent cycles
        agent_tasks = []
        for agent in self.agents.values():
            if agent.state != AgentState.SHUTDOWN:
                task = asyncio.create_task(agent.run_cycle())
                agent_tasks.append(task)
        
        # Wait for all agent cycles to complete
        if agent_tasks:
            await asyncio.gather(*agent_tasks, return_exceptions=True)
    
    async def _process_immediate_tasks(self):
        """Process any immediate system tasks"""
        
        current_time = time.time()
        
        # Health checks
        if current_time - self.last_health_check > self.config.health_check_interval:
            await self._perform_health_check()
            self.last_health_check = current_time
        
        # System optimization
        if current_time - self.last_optimization > self.config.optimization_interval:
            await self._perform_system_optimization()
            self.last_optimization = current_time
    
    async def _message_delivery_loop(self):
        """Background task for message delivery"""
        
        while self.running:
            try:
                delivered = await self.message_bus.deliver_messages()
                self.total_messages_processed += delivered
                
                # Brief pause
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Message delivery error: {e}")
                await asyncio.sleep(1)
    
    async def _system_monitoring_loop(self):
        """Background task for system monitoring"""
        
        while self.running:
            try:
                # Clean up inactive agents
                inactive_count = self.message_bus.cleanup_inactive_agents()
                if inactive_count > 0:
                    self.logger.info(f"Cleaned up {inactive_count} inactive agents")
                
                # Monitor system resources
                await self._monitor_system_resources()
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _state_persistence_loop(self):
        """Background task for state persistence"""
        
        while self.running:
            try:
                current_time = time.time()
                
                if current_time - self.last_state_save > self.config.save_state_interval:
                    await self._save_system_state()
                    self.last_state_save = current_time
                
                # Wait before next save
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"State persistence error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _perform_health_check(self):
        """Perform system health check"""
        
        # Check agent health
        unhealthy_agents = []
        for agent_id, agent in self.agents.items():
            if agent.state == AgentState.ERROR:
                unhealthy_agents.append(agent_id)
        
        if unhealthy_agents:
            self.logger.warning(f"Unhealthy agents detected: {unhealthy_agents}")
        
        # Check message bus health
        bus_stats = self.message_bus.get_statistics()
        queue_size = bus_stats['queue_size']
        
        if queue_size > self.config.message_queue_size * 0.8:
            self.logger.warning(f"Message queue size high: {queue_size}")
        
        # Check hypergraph health
        hg_stats = self.hypergraph.get_statistics()
        node_count = hg_stats['total_nodes']
        
        if node_count > self.config.max_nodes * 0.9:
            self.logger.warning(f"Hypergraph node count high: {node_count}")
            
            # Trigger attention decay
            self.hypergraph.decay_attention(decay_rate=0.05)
    
    async def _perform_system_optimization(self):
        """Perform system optimization"""
        
        # Find autonomy agent
        autonomy_agent = None
        for agent in self.agents.values():
            if agent.agent_type == 'AutonomyAgent':
                autonomy_agent = agent
                break
        
        if autonomy_agent:
            # Request system optimization
            self.message_bus.send_message(
                autonomy_agent.agent_id,
                "system_optimization",
                {'optimization_type': 'comprehensive'},
                priority=0.8
            )
        else:
            # Perform basic optimization
            await self._basic_system_optimization()
    
    async def _basic_system_optimization(self):
        """Basic system optimization when no autonomy agent available"""
        
        # Rebalance attention in hypergraph
        self.hypergraph.decay_attention(decay_rate=self.config.attention_decay_rate)
        
        # Clean up old messages
        self.message_bus.cleanup_inactive_agents()
    
    async def _monitor_system_resources(self):
        """Monitor system resource usage"""
        
        # This would normally check actual system resources
        # For now, we'll just log some basic statistics
        
        stats = {
            'agents': len(self.agents),
            'hypergraph_nodes': len(self.hypergraph.nodes),
            'hypergraph_links': len(self.hypergraph.links),
            'message_queue_size': len(self.message_bus.message_queue),
            'total_messages_processed': self.total_messages_processed,
            'system_errors': self.system_errors,
            'cycles_completed': self.cycles_completed
        }
        
        # Log stats periodically
        if self.cycles_completed % 1000 == 0:
            self.logger.info(f"System stats: {stats}")
    
    async def _save_system_state(self):
        """Save system state to disk"""
        
        try:
            # Save hypergraph
            hg_state = self.hypergraph.to_dict()
            with open("hypergraph_state.json", "w") as f:
                import json
                json.dump(hg_state, f, indent=2)
            
            # Save message bus state
            self.message_bus.save_state("message_bus_state.json")
            
            # Save agent states
            agent_states = {}
            for agent_id, agent in self.agents.items():
                agent_states[agent_id] = agent.get_status()
            
            with open("agent_states.json", "w") as f:
                import json
                json.dump(agent_states, f, indent=2)
            
            self.logger.info("System state saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving system state: {e}")
    
    def _update_system_metrics(self):
        """Update system performance metrics"""
        
        # Update agent metrics
        for agent in self.agents.values():
            agent.last_activity = time.time()
    
    async def submit_goal(self, 
                         goal_name: str,
                         goal_description: str,
                         requirements: List[str] = None,
                         input_data: Dict[str, Any] = None,
                         priority: int = 2) -> Optional[str]:
        """Submit a high-level goal to the system"""
        
        # Find task manager
        task_manager = None
        for agent in self.agents.values():
            if agent.agent_type == 'TaskManagerAgent':
                task_manager = agent
                break
        
        if not task_manager:
            self.logger.error("No TaskManagerAgent available")
            return None
        
        # Submit goal
        goal_data = {
            'name': goal_name,
            'description': goal_description,
            'requirements': requirements or [],
            'input_data': input_data or {},
            'priority': priority
        }
        
        success = self.message_bus.send_message(
            task_manager.agent_id,
            "submit_task",
            goal_data,
            priority=0.7
        )
        
        if success:
            self.logger.info(f"Goal submitted: {goal_name}")
            return goal_name
        else:
            self.logger.error(f"Failed to submit goal: {goal_name}")
            return None
    
    async def query_knowledge(self, 
                             query: str,
                             domain: str = None) -> Optional[Dict[str, Any]]:
        """Query the knowledge hypergraph"""
        
        # Find memory agent
        memory_agent = None
        for agent in self.agents.values():
            if agent.agent_type == 'MemoryAgent':
                memory_agent = agent
                break
        
        if not memory_agent:
            self.logger.error("No MemoryAgent available")
            return None
        
        # Submit query
        query_data = {
            'query': query,
            'domain_filter': domain,
            'context_size': 20
        }
        
        success = self.message_bus.send_message(
            memory_agent.agent_id,
            "retrieve_knowledge",
            query_data,
            priority=0.6
        )
        
        if success:
            self.logger.info(f"Knowledge query submitted: {query}")
            # In a real system, you'd wait for the response
            return {'status': 'query_submitted', 'query': query}
        else:
            self.logger.error(f"Failed to submit query: {query}")
            return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        # Agent status
        agent_status = {}
        for agent_id, agent in self.agents.items():
            agent_status[agent_id] = {
                'type': agent.agent_type,
                'state': agent.state.value,
                'capabilities': agent.capabilities,
                'performance': agent.get_performance_metrics()
            }
        
        # System statistics
        uptime = time.time() - self.startup_time if self.startup_time else 0
        
        return {
            'running': self.running,
            'uptime': uptime,
            'cycles_completed': self.cycles_completed,
            'total_messages_processed': self.total_messages_processed,
            'system_errors': self.system_errors,
            'agents': agent_status,
            'hypergraph_stats': self.hypergraph.get_statistics() if self.hypergraph else {},
            'message_bus_stats': self.message_bus.get_statistics() if self.message_bus else {},
            'background_tasks': len(self.background_tasks)
        }
    
    async def shutdown(self):
        """Gracefully shutdown the cognitive grammar system"""
        
        if not self.running:
            return
        
        self.logger.info("Initiating system shutdown...")
        
        # Stop main loop
        self.running = False
        
        # Shutdown all agents
        shutdown_tasks = []
        for agent in self.agents.values():
            agent.shutdown()
            task = asyncio.create_task(agent.run_cycle())  # Final cycle
            shutdown_tasks.append(task)
        
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Save final state
        await self._save_system_state()
        
        self.logger.info("System shutdown completed")


# Example usage and main entry point
async def main():
    """Main entry point for the cognitive grammar system"""
    
    # Create system configuration
    config = SystemConfiguration()
    config.log_level = logging.INFO
    
    # Create and start system
    system = CognitiveGrammarSystem(config)
    
    try:
        # Initialize system
        if not await system.initialize():
            print("Failed to initialize system")
            return
        
        print("Cognitive Grammar System initialized successfully")
        print("System capabilities:")
        print("- Distributed knowledge representation (hypergraph)")
        print("- Multi-agent cognitive processing")
        print("- Recursive task decomposition and allocation")
        print("- Attention-based resource management")
        print("- Self-modification and optimization")
        print("- Integration with existing neural architectures")
        
        # Example goal submission
        await system.submit_goal(
            goal_name="learn_visual_patterns",
            goal_description="Learn to recognize patterns in visual data",
            requirements=["pattern_recognition", "learning"],
            input_data={"data_type": "visual", "dataset": "example_images"},
            priority=3
        )
        
        # Example knowledge query
        await system.query_knowledge(
            query="visual pattern recognition techniques",
            domain="computer_vision"
        )
        
        # Start the system
        await system.start()
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"System error: {e}")
    finally:
        await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())