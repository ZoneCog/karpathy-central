#!/usr/bin/env python3
"""
Simple Example: Distributed Cognitive Grammar System

This example demonstrates the basic functionality of the distributed
agentic cognitive grammar system, including:

1. System initialization with multiple agents
2. Goal submission and task decomposition  
3. Knowledge storage and retrieval
4. Multi-modal neural processing
5. System monitoring and status reporting

Run with: python examples/simple_example.py
"""

import asyncio
import time
import logging
from pathlib import Path
import sys

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cognitive_grammar import (
    CognitiveGrammarSystem, 
    SystemConfiguration,
    Hypergraph,
    ConceptNode,
    InheritanceLink
)

async def demonstrate_basic_functionality():
    """Demonstrate basic system functionality"""
    
    print("🧠 Cognitive Grammar System - Simple Example")
    print("=" * 50)
    
    # 1. Create and configure system
    print("\n1. Initializing system...")
    config = SystemConfiguration()
    config.log_level = logging.WARNING  # Reduce noise for demo
    config.cycle_interval = 2.0  # Slower cycles for demo
    
    system = CognitiveGrammarSystem(config)
    
    # Initialize system
    if not await system.initialize():
        print("❌ Failed to initialize system")
        return
    
    print("✅ System initialized successfully")
    
    # 2. Start system in background
    print("\n2. Starting system...")
    system_task = asyncio.create_task(system.start())
    
    # Give system time to start
    await asyncio.sleep(3)
    
    # 3. Submit learning goal
    print("\n3. Submitting learning goal...")
    goal_id = await system.submit_goal(
        goal_name="pattern_learning_demo",
        goal_description="Learn patterns from example data for demonstration",
        requirements=["pattern_recognition", "learning"],
        input_data={
            "data_type": "sequences",
            "examples": [[1, 0, 1], [0, 1, 0], [1, 1, 1]],
            "target": "classification"
        },
        priority=3
    )
    
    if goal_id:
        print(f"✅ Goal submitted: {goal_id}")
    else:
        print("❌ Failed to submit goal")
    
    # 4. Query knowledge base
    print("\n4. Querying knowledge base...")
    knowledge_result = await system.query_knowledge(
        query="pattern recognition techniques",
        domain="machine_learning"
    )
    
    if knowledge_result:
        print(f"✅ Knowledge query submitted: {knowledge_result['query']}")
    else:
        print("❌ Failed to query knowledge")
    
    # 5. Demonstrate direct hypergraph interaction
    print("\n5. Adding knowledge to hypergraph...")
    
    # Create some example concepts
    ml_concept = ConceptNode("machine_learning")
    pattern_concept = ConceptNode("pattern_recognition") 
    neural_concept = ConceptNode("neural_networks")
    
    # Add to hypergraph
    system.hypergraph.add_node(ml_concept)
    system.hypergraph.add_node(pattern_concept)
    system.hypergraph.add_node(neural_concept)
    
    # Create relationships
    pattern_is_ml = InheritanceLink(pattern_concept.id, ml_concept.id)
    neural_is_ml = InheritanceLink(neural_concept.id, ml_concept.id)
    
    system.hypergraph.add_link(pattern_is_ml)
    system.hypergraph.add_link(neural_is_ml)
    
    # Spread attention from ML concept
    system.hypergraph.spread_attention(ml_concept.id, intensity=0.8)
    
    print("✅ Added concepts and relationships to hypergraph")
    print(f"   Nodes: {len(system.hypergraph.nodes)}")
    print(f"   Links: {len(system.hypergraph.links)}")
    
    # 6. Monitor system for a while
    print("\n6. Monitoring system status...")
    
    for i in range(5):
        await asyncio.sleep(2)
        
        status = system.get_system_status()
        print(f"\n📊 Status Update {i+1}:")
        print(f"   Uptime: {status['uptime']:.1f}s")
        print(f"   Cycles: {status['cycles_completed']}")
        print(f"   Messages: {status['total_messages_processed']}")
        print(f"   Agents: {len(status['agents'])}")
        print(f"   Knowledge nodes: {status['hypergraph_stats']['total_nodes']}")
        
        # Show agent states
        for agent_id, agent_status in status['agents'].items():
            state = agent_status['state']
            agent_type = agent_status['type']
            tasks = agent_status['performance']['tasks_completed']
            print(f"   {agent_type}: {state} (tasks: {tasks})")
    
    # 7. Demonstrate attention mechanism
    print("\n7. Analyzing attention distribution...")
    
    top_nodes = system.hypergraph.get_most_attended_nodes(5)
    if top_nodes:
        print("🎯 Most attended concepts:")
        for node in top_nodes:
            print(f"   {node.name}: {node.attention_value:.3f}")
    
    # 8. Show final statistics
    print("\n8. Final system statistics...")
    
    final_status = system.get_system_status()
    bus_stats = final_status['message_bus_stats']
    
    print(f"📈 System Performance:")
    print(f"   Total runtime: {final_status['uptime']:.1f}s")
    print(f"   Processing cycles: {final_status['cycles_completed']}")
    print(f"   Messages sent: {bus_stats['messages_sent']}")
    print(f"   Messages delivered: {bus_stats['messages_delivered']}")
    print(f"   Active agents: {bus_stats['active_agents']}")
    print(f"   System errors: {final_status['system_errors']}")
    
    if bus_stats['messages_sent'] > 0:
        delivery_rate = bus_stats['messages_delivered'] / bus_stats['messages_sent']
        print(f"   Message delivery rate: {delivery_rate:.1%}")
    
    # 9. Graceful shutdown
    print("\n9. Shutting down system...")
    await system.shutdown()
    print("✅ System shutdown complete")
    
    print("\n🎉 Demo completed successfully!")
    print("\nWhat happened:")
    print("• Multiple intelligent agents were created and coordinated")
    print("• Tasks were decomposed and distributed automatically") 
    print("• Knowledge was stored in a hypergraph with attention mechanisms")
    print("• The system demonstrated self-organization and resource allocation")
    print("• All subsystems communicated through a unified message bus")

async def demonstrate_advanced_features():
    """Demonstrate advanced system features"""
    
    print("\n🚀 Advanced Features Demo")
    print("=" * 30)
    
    # Create system with custom configuration
    config = SystemConfiguration()
    config.memory_agents = 2  # Multiple memory agents
    config.cognitive_agents = 2  # Multiple cognitive agents
    config.enable_scriptbots = True
    config.enable_recurrentjs = True
    config.log_level = logging.ERROR  # Minimal logging
    
    system = CognitiveGrammarSystem(config)
    await system.initialize()
    
    # Start system
    system_task = asyncio.create_task(system.start())
    await asyncio.sleep(2)
    
    print("✅ Advanced system initialized")
    
    # Submit complex multi-modal goal
    await system.submit_goal(
        goal_name="multimodal_analysis",
        goal_description="Analyze data across multiple modalities",
        requirements=["multimodal_processing", "pattern_recognition", "reasoning"],
        input_data={
            "modalities": ["visual", "text", "sequence"],
            "data": {
                "image": "example.jpg",
                "text": "Analyze this complex pattern",
                "sequence": [1, 2, 3, 4, 5]
            }
        },
        priority=4  # Critical priority
    )
    
    print("✅ Complex multimodal goal submitted")
    
    # Monitor for shorter time
    for i in range(3):
        await asyncio.sleep(2)
        status = system.get_system_status()
        print(f"📊 Advanced Status {i+1}: {status['cycles_completed']} cycles, "
              f"{status['total_messages_processed']} messages")
    
    # Shutdown
    await system.shutdown()
    print("✅ Advanced demo complete")

async def main():
    """Main demo function"""
    
    try:
        # Run basic demonstration
        await demonstrate_basic_functionality()
        
        # Optional: Run advanced features
        print("\n" + "="*60)
        run_advanced = input("\nRun advanced features demo? (y/N): ").lower().startswith('y')
        
        if run_advanced:
            await demonstrate_advanced_features()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nThank you for exploring the Cognitive Grammar System! 🧠✨")

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())