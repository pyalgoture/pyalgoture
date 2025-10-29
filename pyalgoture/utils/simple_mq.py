"""Simple message queue implementation using Redis for pub/sub and FIFO operations."""

import json
import re
import secrets
import string
import sys
from collections.abc import Callable
from importlib import import_module
from typing import Any

try:
    from redis import Redis
except ImportError:
    Redis = Any


def is_json_string(data: str) -> dict[str, Any]:
    """Parse a string as JSON, handling common formatting issues.

    Args:
        data: String that might be JSON format

    Returns:
        Parsed JSON dictionary, or empty dict if parsing fails
    """
    res = {}
    try:
        # Fix common JSON formatting issues
        normalized_data = re.sub(r"'?([\w-]+)'?\s*:", r'"\1":', data)
        normalized_data = re.sub(r":\s*'(.*?)'", r':"\1"', normalized_data)
        res = json.loads(normalized_data)
    except (ValueError, json.JSONDecodeError):
        pass
    return res


class SignalMQ:
    """Simple FIFO message queue using Redis for both queue and pub/sub operations.

    This class provides both traditional queue operations (enqueue/dequeue) and
    pub/sub messaging capabilities using Redis as the backend.
    """

    REDIS_KEY_PREFIX = "PYALGOTURE_SIGNAL_MQ"
    RANDOM_NAME_LENGTH = 16

    def __init__(
        self,
        conn: Redis | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize SignalMQ with Redis connection and queue name.

        Args:
            conn: Redis connection object. If None, creates default connection
            name: Queue name. If None, generates random name

        Raises:
            SystemExit: If Redis package is not available
        """
        self._conn = self._setup_connection(conn)
        self._name = self._setup_name(name)
        self._pubsub = self._conn.pubsub()
        self._thread = self._pubsub.run_in_thread(sleep_time=0.001, daemon=True)

    def _setup_connection(self, conn: Redis | None) -> Redis:
        """Setup Redis connection."""
        if conn is not None:
            return conn

        try:
            redis_module = import_module("redis")
            return redis_module.Redis()
        except ImportError:
            print("ERROR: SignalMQ requires 'redis' package. Install with: pip install redis")
            sys.exit(1)

    def _setup_name(self, name: str | None) -> str:
        """Setup queue name."""
        if name is not None:
            return str(name)

        # Generate random name
        chars = string.ascii_lowercase + string.digits
        return "".join(secrets.choice(chars) for _ in range(self.RANDOM_NAME_LENGTH))

    def _redis_key(self) -> str:
        """Get the full Redis key for this queue."""
        return f"{self.REDIS_KEY_PREFIX}_{self.name}"

    @property
    def conn(self) -> Redis:
        """Redis connection object."""
        return self._conn

    @conn.setter
    def conn(self, new_conn: Redis) -> None:
        """Set new Redis connection."""
        self._conn = new_conn

    @property
    def name(self) -> str:
        """Queue name."""
        return self._name

    @name.setter
    def name(self, new_name: str) -> None:
        """Set new queue name."""
        self._name = str(new_name)

    @property
    def is_connected(self) -> bool:
        """Check if Redis connection is active."""
        res = False
        try:
            res = self._conn.ping()
        except Exception:
            pass
        return res

    def size(self) -> int:
        """Get current number of messages in the queue.

        Returns:
            Number of messages in queue
        """
        return int(self.conn.llen(self._redis_key()))

    def clear(self) -> int:
        """Delete all messages from the queue.

        Returns:
            Number of messages that were cleared
        """
        total = 0
        while True:
            message = self.dequeue()
            if message is None:
                break
            total += 1
        return total

    def enqueue(self, message: str) -> int:
        """Add one message to the end of the queue.

        Args:
            message: Message to add to queue

        Returns:
            Size of queue after enqueuing
        """
        return int(self.conn.rpush(self._redis_key(), str(message)))

    def enqueue_bulk(self, messages: list[str]) -> int:
        """Add multiple messages to the queue at once.

        Args:
            messages: List of messages to add

        Returns:
            Size of queue after enqueuing all messages
        """
        if not messages:
            return self.size()

        # Use pipeline for better performance
        pipe = self.conn.pipeline()
        for message in messages:
            pipe.rpush(self._redis_key(), str(message))
        results = pipe.execute()

        # Return the final queue size
        return results[-1] if results else self.size()

    def dequeue(self) -> str | None:
        """Remove and return one message from the front of the queue.

        Returns:
            Message string, or None if queue is empty
        """
        value = self.conn.lpop(self._redis_key())
        return value.decode("utf-8") if value is not None else None

    def dequeue_bulk(self, max_count: int | None = None) -> list[str]:
        """Remove and return multiple messages from the queue.

        Args:
            max_count: Maximum number of messages to dequeue.
                      If None, dequeues all messages

        Returns:
            List of message strings (empty if queue is empty)

        Raises:
            ValueError: If max_count is negative
        """
        if max_count is not None and max_count < 0:
            raise ValueError("max_count cannot be negative")

        messages = []
        count = 0

        while max_count is None or count < max_count:
            message = self.dequeue()
            if message is None:
                break
            messages.append(message)
            count += 1

        return messages

    def publish(self, message: str | dict[str, Any] | Any) -> None:
        """Publish a message to subscribers.

        Args:
            message: Message to publish. Dicts are JSON-serialized,
                    other types are converted to strings
        """
        if isinstance(message, dict):
            serialized_message = json.dumps(message, default=str)
        elif isinstance(message, str):
            serialized_message = message
        else:
            serialized_message = str(message)

        self._conn.publish(self._redis_key(), serialized_message)

    def subscribe(self, callback: Callable[[Any], None]) -> None:
        """Subscribe to messages with a callback function.

        Args:
            callback: Function to call when messages are received.
                     Receives the parsed message as argument
        """

        def message_handler(redis_message: dict[str, Any]) -> None:
            """Process incoming Redis pub/sub messages."""
            if redis_message["type"] != "message" or redis_message["channel"].decode("utf-8") != self._redis_key():
                return

            message_data = redis_message["data"].decode("utf-8")

            # Try to parse as JSON if it looks like JSON
            if self._looks_like_json(message_data):
                try:
                    parsed_message = json.loads(message_data)
                except json.JSONDecodeError:
                    parsed_message = message_data
            else:
                parsed_message = message_data

            if parsed_message:
                callback(parsed_message)

        self._pubsub.subscribe(**{self._redis_key(): message_handler})

    def _looks_like_json(self, data: str) -> bool:
        """Check if string looks like JSON data."""
        return "{" in data and "}" in data

    def close(self) -> None:
        """Close the pub/sub connection and stop the background thread."""
        if hasattr(self, "_thread") and self._thread:
            self._thread.stop()
        if hasattr(self, "_pubsub") and self._pubsub:
            self._pubsub.close()


if __name__ == "__main__":
    """Example usage of SignalMQ."""
    import time
    from datetime import datetime

    # Create queue instance
    mq = SignalMQ(name="signal_348444")

    # Example 1: Queue operations
    msg = f"Hello, World! @ {datetime.now()}"
    mq.enqueue(msg)
    print(f"Enqueued: {msg}")

    dequeued_msg = mq.dequeue()
    print(f"Dequeued: {dequeued_msg}")

    # Example 2: Pub/Sub operations
    signal_data = {
        "asset_type": "FUTURE",
        "commission": -0.008364,
        "commission_asset": "USDT",
        "datetime": "2023-01-12 15:03:20.523000+08:00",
        "exchange": "BYBIT",
        "name": "BTCUSDT",
        "order_id": "22427-15320-337-00001",
        "ori_order_id": 12989343320,
        "reix_bonus": 0,
        "side": "BUY",
        "leverage": 20,
        "price": "4.480",
        "quantity": "14.051",
        "symbol": "APEUSDT",
    }

    # Publish signal
    mq.publish(signal_data)
    print("Published signal data")

    # Subscribe example (commented out to avoid blocking)
    # def callback(msg):
    #     print(f"Received: {msg}")
    #
    # mq.subscribe(callback=callback)
    # while True:
    #     time.sleep(1)

    # Clean up
    mq.close()
