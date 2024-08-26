import asyncio
import heapq
from typing import List, Callable, Tuple

from event.event_class import EventClass
from event.get_person_in_room_event import GetPersonInRoomEvent

class EventBus:
    # List to store concurrent and sequential commands (async functions)
    concurrent_cmd: List[Callable[[], asyncio.Future]] = []
    sequential_cmd: List[Tuple[int, Callable[[], asyncio.Future]]] = []
    running_tasks: List[asyncio.Task] = []

    # Set up new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    @staticmethod
    def add_concurrent_cmd(cmd: Callable[[], asyncio.Future]) -> None:
        EventBus.concurrent_cmd.append(cmd)

    @staticmethod
    def add_sequential_cmd(cmd: Callable[[], asyncio.Future], priority: int) -> None:
        # Use a heap queue to maintain a priority queue
        heapq.heappush(EventBus.sequential_cmd, (priority, cmd))

    @staticmethod
    def stop_concurrent_cmds() -> None:
        for task in EventBus.running_tasks:
            if not task.done():
                task.cancel()
        EventBus.running_tasks.clear()

    @staticmethod
    def publish(event: 'EventClass') -> None:
        loop = asyncio.get_event_loop()
        for cmd, mode, priority in event.get_cmds():
            if mode == "concurrent":
                EventBus.add_concurrent_cmd(cmd)
            elif mode == "sequential":
                EventBus.add_sequential_cmd(cmd, priority)

        try:
            print("Executing concurrent commands:")
            EventBus.running_tasks = [loop.create_task(cmd()) for cmd in EventBus.concurrent_cmd]

            loop.run_forever()
            EventBus.stop_concurrent_cmds()
            # Example of stopping concurrent commands
            print("\nStopping concurrent commands")
            
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            print("Done")

# Example usage
getPersonInRoomEvent = GetPersonInRoomEvent()
EventBus.publish(getPersonInRoomEvent)

