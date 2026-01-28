# Streaming Processing Abstraction Layer
# Designed for Python workers today, Flink migration tomorrow

from src.streaming.interfaces import (
    KeyedProcessor,
    ProcessorContext,
    SessionWindowAssigner,
    SlidingWindowAssigner,
    StreamProcessor,
    TimeWindow,
    TumblingWindowAssigner,
    WindowAssigner,
    WindowedProcessor,
)
from src.streaming.runtime import (
    DataStream,
    KeyedStream,
    StreamExecutionEnvironment,
)
from src.streaming.serde import (
    AvroSerde,
    Deserializer,
    JsonSerde,
    Serializer,
)
from src.streaming.state import (
    ListState,
    MapState,
    ReducingState,
    StateBackend,
    ValueState,
)

__all__ = [
    # Processors
    "StreamProcessor",
    "WindowedProcessor",
    "KeyedProcessor",
    "ProcessorContext",
    # Windows
    "TimeWindow",
    "WindowAssigner",
    "TumblingWindowAssigner",
    "SlidingWindowAssigner",
    "SessionWindowAssigner",
    # State
    "StateBackend",
    "ValueState",
    "ListState",
    "MapState",
    "ReducingState",
    # Serialization
    "Serializer",
    "Deserializer",
    "JsonSerde",
    "AvroSerde",
    # Runtime
    "StreamExecutionEnvironment",
    "DataStream",
    "KeyedStream",
]
