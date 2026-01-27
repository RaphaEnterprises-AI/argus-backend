# Streaming Processing Abstraction Layer
# Designed for Python workers today, Flink migration tomorrow

from src.streaming.interfaces import (
    StreamProcessor,
    WindowedProcessor,
    KeyedProcessor,
    ProcessorContext,
    TimeWindow,
    WindowAssigner,
    TumblingWindowAssigner,
    SlidingWindowAssigner,
    SessionWindowAssigner,
)
from src.streaming.state import (
    StateBackend,
    ValueState,
    ListState,
    MapState,
    ReducingState,
)
from src.streaming.serde import (
    Serializer,
    Deserializer,
    JsonSerde,
    AvroSerde,
)
from src.streaming.runtime import (
    StreamExecutionEnvironment,
    DataStream,
    KeyedStream,
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
