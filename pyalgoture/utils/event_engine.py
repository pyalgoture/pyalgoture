from collections import defaultdict
from collections.abc import Callable
from heapq import heappop, heappush
from queue import Empty, Queue
from threading import Thread
from time import sleep, time
from typing import Any

"""
priority (higher number has higher priority)
EVENT_TICK = 1
EVENT_LOG = 2
EVENT_TIMER = 2
EVENT_BAR = 3

EVENT_POSITION = 11
EVENT_TRADE = 12
EVENT_ORDER = 13
EVENT_ACCOUNT = 14

"""
EVENT_LOG = "Log"
EVENT_TIMER = "Timer"
EVENT_TICK = "Tick"
EVENT_BAR = "Bar"
EVENT_TRADE = "Trade"
EVENT_ORDER = "Order"
EVENT_POSITION = "Position"
EVENT_ACCOUNT = "Account"

EVENT_CONTRACT = "Contract"
EVENT_BACKTEST = "Backtest"


class PriorityQueue(Queue):
    """Variant of Queue that retrieves open entries in priority order (lowest first).

    Entries are typically tuples of the form:  (priority number, data).

    use heaps(Aka. Min Heap), this implementation push the item in the form of (priority, index, item),
    when pop the item, it will pop the item with the lowest number(highest priority), and then looking for the lowest index (in order to achieve FIFO)
    """

    def _init(self, maxsize: int) -> None:
        self.queue: list = []  # create a list to store item
        self._index: int = 0  # create a index used to record the sequence of push item

    def _qsize(self) -> int:
        return len(self.queue)

    def _put(self, item: Any) -> None:
        """
        the queue consist of a list of set() in (priority, index, item)
        """
        heappush(self.queue, (-item[0], self._index, item[1]))
        self._index += 1

    def _get(self) -> Any:
        return heappop(self.queue)[-1]  # return the item with the highest priority


class Event:
    """
    Event object consists of a type string which is used
    by event engine for distributing event, and a data
    object which contains the real data.
    """

    def __init__(self, type: str, data: Any = None, priority: int = 1) -> None:
        """"""
        self.type: str = type
        self.data: Any = data
        self.priority: int = priority


# Defines handler function to be used in event engine.
HandlerType = Callable[[Event], None]


class EventEngine:
    """
    Event engine distributes event object based on its type
    to those handlers registered.
    It also generates timer event by every interval seconds,
    which can be used for timing purpose.
    """

    def __init__(self, interval: int = 1, debug: bool = False) -> None:
        """
        Timer event is generated every 1 second by default, if
        interval not specified.
        """
        self._interval: int = interval
        self._queue: PriorityQueue = PriorityQueue()
        self._active: bool = False
        self._thread: Thread = Thread(target=self._run)
        self._timer: Thread = Thread(target=self._run_timer)
        self._handlers: defaultdict[str, list[HandlerType]] = defaultdict(list)
        self._events: set[str] = set()
        self._debug = debug
        # self._general_handlers: list = []

    def _run(self) -> None:
        """
        Get event from queue and then process it.
        """
        while self._active:
            try:
                event: Event = self._queue.get(block=True, timeout=1)
                if self._debug:
                    print(
                        f"[DEBUG{int(time())} - run] event:{event.type} - len:{self._queue.qsize()} - data:{event.data}"
                    )
                    st = time()
                    self._process(event)
                    print(f"[DEBUG{int(time())} - processed] event:{event.type} time consumed:{time() - st}s")
                else:
                    self._process(event)

            except Empty:
                pass

    def _process(self, event: Event) -> None:
        """
        First distribute event to those handlers registered listening
        to this type.
        Then distribute event to those general handlers which listens
        to all types.
        """
        if event.type in self._handlers:
            [handler(event) for handler in self._handlers[event.type]]

    def _run_timer(self) -> None:
        """
        Sleep by interval second(s) and then generate a timer event.
        """
        while self._active:
            sleep(self._interval)
            event: Event = Event(type=EVENT_TIMER, priority=2)
            self.put(event)

    @property
    def empty(self) -> bool:
        return self._queue.empty()

    @property
    def size(self) -> int:
        return self._queue.qsize()

    @property
    def full(self) -> bool:
        return self._queue.full()

    @property
    def status(self) -> bool:
        return self._active

    @property
    def events(self) -> set[str]:
        return self._events

    @property
    def debug(self) -> bool:
        return self._debug

    @debug.setter
    def debug(self, new_debug: bool) -> None:
        self._debug = new_debug

    def start(self) -> None:
        """
        Start event engine to process events and generate timer events.
        """
        if not self._active:
            self._active = True
            self._thread.start()
            self._timer.start()

    def stop(self) -> None:
        """
        Stop event engine.
        """
        if self._active:
            self._active = False
            self._timer.join()
            self._thread.join()

    def close(self) -> None:
        self.stop()

    def put(self, event: Event) -> None:
        """
        Put an event object into event queue.
        """
        if self._debug:
            print(f"[DEBUG{int(time())} - put] event:{event.type} - len:{self._queue.qsize()} - data:{event.data}")
        self._queue.put([event.priority, event])

    def register(self, type: str, handler: HandlerType) -> None:
        """
        Register a new handler function for a specific event type. Every
        function can only be registered once for each event type.
        """
        handler_list: list[HandlerType] = self._handlers[type]
        if handler not in handler_list:
            self._events.add(type)
            handler_list.append(handler)
            print(f"{type} is registered.")

    def unregister(self, type: str, handler: HandlerType) -> None:
        """
        Unregister an existing handler function from event engine.
        """
        handler_list: list[HandlerType] = self._handlers[type]

        if handler in handler_list:
            handler_list.remove(handler)

        if not handler_list:
            if type in self._events:
                self._events.remove(type)
            self._handlers.pop(type, None)
            print(f"{type} is unregistered.")


# if __name__ == "__main__":
#     import time
#     from datetime import datetime

#     def simpletest(event):
#         print(f"prceoss timer event callback: {str(datetime.now())} - event type:{event.type} - data: {event.data}")

#     def simpletest1(event):
#         print(f"[11111]prceoss timer event callback: {str(datetime.now())} - event type:{event.type} - data: {event.data}")

#     def tickevent(event):
#         while not event_engine.empty:
#             print(f"        TT ignore tick: {event.data}")
#             return
#         print(f"tick dat: {str(datetime.now())} - event type:{event.type} - data: {event.data}")
#         time.sleep(0.5)
#         event.data["now"] = datetime.now()
#         event_engine.put(Event("tick_callback", data=event.data))

#     def tick_callbackevent(event):
#         print(f"--> callback: {str(datetime.now())} - event type:{event.type} - data: {event.data}")
#         time.sleep(1)

#     event_engine = EventEngine()
#     event_engine.register(EVENT_TIMER, simpletest)
#     # event_engine.register(EVENT_TIMER, simpletest1)
#     event_engine.register(EVENT_TICK, tickevent)
#     event_engine.register("tick_callback", tick_callbackevent)
#     event_engine.start()

#     time.sleep(2)
#     for i in range(30):
#         time.sleep(0.1)
#         event_engine.put(Event(EVENT_TICK, data={"o": 1, "c": i, "h": 3, "l": -1}, priority=2))
#     # time.sleep(2)
#     # event_engine.unregister(EVENT_TICK, tickevent)
#     time.sleep(2)
#     event_engine.put(Event(EVENT_TICK, data={"o": 11, "c": 12, "h": 13, "l": -1}, priority=2))

#     time.sleep(5)
#     event_engine.stop()
