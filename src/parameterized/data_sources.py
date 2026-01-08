"""Data source handlers for parameterized testing.

This module provides various data source implementations for loading
test parameter data from different sources:
- InlineDataSource: Direct values in test specification
- CSVDataSource: Load from CSV files
- JSONDataSource: Load from JSON files
- EnvironmentDataSource: Load from environment variables

Each data source implements the BaseDataSource interface and can be
created using the DataSourceFactory.
"""

import csv
import json
import os
import re
from abc import ABC, abstractmethod
from io import StringIO
from pathlib import Path
from typing import Any, Optional

import structlog

from src.parameterized.models import DataSource, DataSourceType, ParameterSet

logger = structlog.get_logger()


class DataSourceError(Exception):
    """Exception raised when data source operations fail."""

    pass


class BaseDataSource(ABC):
    """Abstract base class for data sources.

    All data source implementations must inherit from this class and
    implement the load() method to retrieve parameter sets.

    Attributes:
        config: DataSource configuration
        _cache: Cached parameter sets (for expensive load operations)
    """

    def __init__(self, config: DataSource):
        """Initialize the data source.

        Args:
            config: DataSource configuration object
        """
        self.config = config
        self._cache: Optional[list[ParameterSet]] = None

    @abstractmethod
    def load(self) -> list[ParameterSet]:
        """Load parameter sets from the data source.

        Returns:
            List of ParameterSet objects

        Raises:
            DataSourceError: If loading fails
        """
        pass

    def reload(self) -> list[ParameterSet]:
        """Force reload of parameter sets, clearing cache.

        Returns:
            List of ParameterSet objects
        """
        self._cache = None
        return self.load()

    def validate(self) -> list[str]:
        """Validate the data source configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        return []

    def _apply_mapping(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply field mapping to data if configured.

        Args:
            data: Raw data dictionary

        Returns:
            Mapped data dictionary
        """
        if not self.config.mapping:
            return data

        mapped = {}
        for source_field, target_field in self.config.mapping.items():
            if source_field in data:
                mapped[target_field] = data[source_field]

        # Include unmapped fields
        for key, value in data.items():
            if key not in self.config.mapping:
                mapped[key] = value

        return mapped

    def _apply_filter(self, data_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply filter expression to data if configured.

        Supports simple expressions like:
        - "field == 'value'"
        - "field != 'value'"
        - "field > 10"
        - "field in ['a', 'b']"

        Args:
            data_list: List of data dictionaries

        Returns:
            Filtered list of data dictionaries
        """
        if not self.config.filter:
            return data_list

        filter_expr = self.config.filter.strip()
        filtered = []

        for data in data_list:
            try:
                # Create a safe evaluation context with only the data fields
                context = {k: v for k, v in data.items()}
                # Evaluate the filter expression
                if self._evaluate_filter(filter_expr, context):
                    filtered.append(data)
            except Exception as e:
                logger.warning(
                    "Filter evaluation failed for data item",
                    filter=filter_expr,
                    error=str(e),
                )

        return filtered

    def _evaluate_filter(self, expr: str, context: dict[str, Any]) -> bool:
        """Safely evaluate a filter expression.

        Args:
            expr: Filter expression string
            context: Dictionary of values for evaluation

        Returns:
            Boolean result of evaluation
        """
        # Parse simple comparison expressions
        patterns = [
            (r"(\w+)\s*==\s*'([^']*)'", lambda m, c: c.get(m.group(1)) == m.group(2)),
            (r"(\w+)\s*==\s*\"([^\"]*)\"", lambda m, c: c.get(m.group(1)) == m.group(2)),
            (r"(\w+)\s*==\s*(\d+)", lambda m, c: c.get(m.group(1)) == int(m.group(2))),
            (r"(\w+)\s*!=\s*'([^']*)'", lambda m, c: c.get(m.group(1)) != m.group(2)),
            (r"(\w+)\s*!=\s*\"([^\"]*)\"", lambda m, c: c.get(m.group(1)) != m.group(2)),
            (r"(\w+)\s*>\s*(\d+)", lambda m, c: (c.get(m.group(1)) or 0) > int(m.group(2))),
            (r"(\w+)\s*<\s*(\d+)", lambda m, c: (c.get(m.group(1)) or 0) < int(m.group(2))),
            (r"(\w+)\s*>=\s*(\d+)", lambda m, c: (c.get(m.group(1)) or 0) >= int(m.group(2))),
            (r"(\w+)\s*<=\s*(\d+)", lambda m, c: (c.get(m.group(1)) or 0) <= int(m.group(2))),
            (
                r"(\w+)\s+in\s+\[([^\]]+)\]",
                lambda m, c: c.get(m.group(1)) in self._parse_list(m.group(2)),
            ),
        ]

        for pattern, evaluator in patterns:
            match = re.match(pattern, expr)
            if match:
                return evaluator(match, context)

        # If no pattern matches, try simple truthiness check
        field_match = re.match(r"(\w+)", expr)
        if field_match:
            return bool(context.get(field_match.group(1)))

        return True

    def _parse_list(self, list_str: str) -> list[Any]:
        """Parse a string representation of a list.

        Args:
            list_str: String like "'a', 'b', 'c'" or "1, 2, 3"

        Returns:
            Parsed list
        """
        items = []
        for item in list_str.split(","):
            item = item.strip()
            if item.startswith("'") and item.endswith("'"):
                items.append(item[1:-1])
            elif item.startswith('"') and item.endswith('"'):
                items.append(item[1:-1])
            elif item.isdigit():
                items.append(int(item))
            else:
                items.append(item)
        return items

    def _apply_limit(self, data_list: list[ParameterSet]) -> list[ParameterSet]:
        """Apply limit to parameter sets if configured.

        Args:
            data_list: List of ParameterSet objects

        Returns:
            Limited list of ParameterSet objects
        """
        if self.config.limit:
            return data_list[: self.config.limit]
        return data_list

    def _create_parameter_set(
        self,
        data: dict[str, Any],
        index: int,
        name_template: Optional[str] = None,
    ) -> ParameterSet:
        """Create a ParameterSet from data dictionary.

        Args:
            data: Data dictionary
            index: Index of this parameter set
            name_template: Optional name template with {{field}} placeholders

        Returns:
            ParameterSet object
        """
        # Generate name from template or data
        if name_template:
            name = name_template
            for key, value in data.items():
                name = name.replace(f"{{{{{key}}}}}", str(value))
        elif "name" in data:
            name = str(data.pop("name"))
        elif "_name" in data:
            name = str(data.pop("_name"))
        else:
            # Generate name from first few values
            preview = "_".join(str(v)[:20] for v in list(data.values())[:3])
            name = f"case_{index}_{preview}"

        # Extract description if present
        description = data.pop("description", None) or data.pop("_description", None)

        # Extract tags if present
        tags = data.pop("tags", None) or data.pop("_tags", None) or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]

        # Extract skip info if present
        skip = data.pop("skip", False) or data.pop("_skip", False)
        skip_reason = data.pop("skip_reason", None) or data.pop("_skip_reason", None)

        return ParameterSet(
            name=name,
            values=data,
            description=description,
            tags=tags,
            skip=bool(skip),
            skip_reason=skip_reason,
        )


class InlineDataSource(BaseDataSource):
    """Data source for inline parameter values.

    Loads parameter sets directly from the data field in the configuration.

    Example config:
        {
            "type": "inline",
            "data": [
                {"username": "admin", "password": "admin123"},
                {"username": "user", "password": "user123"}
            ]
        }
    """

    def load(self) -> list[ParameterSet]:
        """Load parameter sets from inline data.

        Returns:
            List of ParameterSet objects

        Raises:
            DataSourceError: If data is not provided
        """
        if self._cache is not None:
            return self._cache

        if not self.config.data:
            raise DataSourceError("Inline data source requires 'data' field")

        logger.debug("Loading inline data", count=len(self.config.data))

        # Apply mapping if configured
        data_list = [self._apply_mapping(d.copy()) for d in self.config.data]

        # Apply filter if configured
        data_list = self._apply_filter(data_list)

        # Create parameter sets
        parameter_sets = [
            self._create_parameter_set(data.copy(), i)
            for i, data in enumerate(data_list)
        ]

        # Apply limit
        parameter_sets = self._apply_limit(parameter_sets)

        self._cache = parameter_sets
        return parameter_sets

    def validate(self) -> list[str]:
        """Validate inline data source configuration.

        Returns:
            List of validation errors
        """
        errors = []

        if not self.config.data:
            errors.append("Inline data source requires 'data' field")
        elif not isinstance(self.config.data, list):
            errors.append("Inline data must be a list")
        elif len(self.config.data) == 0:
            errors.append("Inline data cannot be empty")
        else:
            for i, item in enumerate(self.config.data):
                if not isinstance(item, dict):
                    errors.append(f"Data item at index {i} must be a dictionary")

        return errors


class CSVDataSource(BaseDataSource):
    """Data source for CSV file parameter values.

    Loads parameter sets from a CSV file. The first row is treated as
    headers (parameter names) unless 'has_header' is False.

    Example config:
        {
            "type": "csv",
            "path": "test_data/users.csv",
            "mapping": {"user_name": "username"},
            "delimiter": ","
        }
    """

    def load(self) -> list[ParameterSet]:
        """Load parameter sets from CSV file.

        Returns:
            List of ParameterSet objects

        Raises:
            DataSourceError: If file cannot be read
        """
        if self._cache is not None:
            return self._cache

        if not self.config.path:
            raise DataSourceError("CSV data source requires 'path' field")

        path = Path(self.config.path)
        if not path.exists():
            raise DataSourceError(f"CSV file not found: {path}")

        logger.debug("Loading CSV data", path=str(path))

        try:
            with open(path, "r", encoding=self.config.encoding) as f:
                reader = csv.DictReader(f, delimiter=self.config.delimiter)
                data_list = list(reader)
        except Exception as e:
            raise DataSourceError(f"Failed to read CSV file: {e}")

        # Apply mapping
        data_list = [self._apply_mapping(d) for d in data_list]

        # Apply filter
        data_list = self._apply_filter(data_list)

        # Create parameter sets
        parameter_sets = [
            self._create_parameter_set(data.copy(), i)
            for i, data in enumerate(data_list)
        ]

        # Apply limit
        parameter_sets = self._apply_limit(parameter_sets)

        self._cache = parameter_sets
        logger.info("Loaded CSV data", path=str(path), count=len(parameter_sets))
        return parameter_sets

    def load_from_string(self, csv_content: str) -> list[ParameterSet]:
        """Load parameter sets from CSV string content.

        Args:
            csv_content: CSV content as string

        Returns:
            List of ParameterSet objects
        """
        reader = csv.DictReader(StringIO(csv_content), delimiter=self.config.delimiter)
        data_list = list(reader)

        # Apply mapping and filter
        data_list = [self._apply_mapping(d) for d in data_list]
        data_list = self._apply_filter(data_list)

        # Create parameter sets
        parameter_sets = [
            self._create_parameter_set(data.copy(), i)
            for i, data in enumerate(data_list)
        ]

        return self._apply_limit(parameter_sets)

    def validate(self) -> list[str]:
        """Validate CSV data source configuration.

        Returns:
            List of validation errors
        """
        errors = []

        if not self.config.path:
            errors.append("CSV data source requires 'path' field")
        else:
            path = Path(self.config.path)
            if not path.exists():
                errors.append(f"CSV file not found: {path}")
            elif not path.is_file():
                errors.append(f"Path is not a file: {path}")
            elif path.suffix.lower() not in [".csv", ".tsv"]:
                errors.append(f"File does not appear to be a CSV: {path}")

        return errors


class JSONDataSource(BaseDataSource):
    """Data source for JSON file parameter values.

    Loads parameter sets from a JSON file. The file should contain
    an array of objects, or an object with a key containing the array.

    Example config:
        {
            "type": "json",
            "path": "test_data/users.json"
        }

    JSON file formats supported:
        1. Array of objects: [{"username": "admin"}, {"username": "user"}]
        2. Object with array: {"users": [{"username": "admin"}]}
    """

    def load(self) -> list[ParameterSet]:
        """Load parameter sets from JSON file.

        Returns:
            List of ParameterSet objects

        Raises:
            DataSourceError: If file cannot be read or parsed
        """
        if self._cache is not None:
            return self._cache

        if not self.config.path:
            raise DataSourceError("JSON data source requires 'path' field")

        path = Path(self.config.path)
        if not path.exists():
            raise DataSourceError(f"JSON file not found: {path}")

        logger.debug("Loading JSON data", path=str(path))

        try:
            with open(path, "r", encoding=self.config.encoding) as f:
                content = json.load(f)
        except json.JSONDecodeError as e:
            raise DataSourceError(f"Invalid JSON in file: {e}")
        except Exception as e:
            raise DataSourceError(f"Failed to read JSON file: {e}")

        # Handle different JSON structures
        if isinstance(content, list):
            data_list = content
        elif isinstance(content, dict):
            # Look for an array in the object
            data_list = None
            for key, value in content.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], dict):
                        data_list = value
                        break

            if data_list is None:
                raise DataSourceError(
                    "JSON object must contain an array of objects"
                )
        else:
            raise DataSourceError(
                "JSON must be an array or object containing an array"
            )

        # Apply mapping
        data_list = [self._apply_mapping(d) for d in data_list]

        # Apply filter
        data_list = self._apply_filter(data_list)

        # Create parameter sets
        parameter_sets = [
            self._create_parameter_set(data.copy(), i)
            for i, data in enumerate(data_list)
        ]

        # Apply limit
        parameter_sets = self._apply_limit(parameter_sets)

        self._cache = parameter_sets
        logger.info("Loaded JSON data", path=str(path), count=len(parameter_sets))
        return parameter_sets

    def load_from_string(self, json_content: str) -> list[ParameterSet]:
        """Load parameter sets from JSON string content.

        Args:
            json_content: JSON content as string

        Returns:
            List of ParameterSet objects
        """
        content = json.loads(json_content)

        if isinstance(content, list):
            data_list = content
        elif isinstance(content, dict):
            # Look for array in object
            data_list = None
            for key, value in content.items():
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    data_list = value
                    break
            if data_list is None:
                raise DataSourceError("JSON must contain an array of objects")
        else:
            raise DataSourceError("Invalid JSON structure")

        # Apply mapping and filter
        data_list = [self._apply_mapping(d) for d in data_list]
        data_list = self._apply_filter(data_list)

        # Create parameter sets
        parameter_sets = [
            self._create_parameter_set(data.copy(), i)
            for i, data in enumerate(data_list)
        ]

        return self._apply_limit(parameter_sets)

    def validate(self) -> list[str]:
        """Validate JSON data source configuration.

        Returns:
            List of validation errors
        """
        errors = []

        if not self.config.path:
            errors.append("JSON data source requires 'path' field")
        else:
            path = Path(self.config.path)
            if not path.exists():
                errors.append(f"JSON file not found: {path}")
            elif not path.is_file():
                errors.append(f"Path is not a file: {path}")

        return errors


class EnvironmentDataSource(BaseDataSource):
    """Data source for environment variable parameter values.

    Loads parameter values from environment variables. Can use either
    a prefix to find variables or explicit mapping.

    Example config with prefix:
        {
            "type": "env",
            "env_prefix": "TEST_USER_"
        }
        Loads: TEST_USER_USERNAME -> username, TEST_USER_PASSWORD -> password

    Example config with mapping:
        {
            "type": "env",
            "env_mapping": {
                "MY_USERNAME": "username",
                "MY_PASSWORD": "password"
            }
        }
    """

    def load(self) -> list[ParameterSet]:
        """Load parameter sets from environment variables.

        Returns:
            List of ParameterSet objects (single set for env source)

        Raises:
            DataSourceError: If configuration is invalid
        """
        if self._cache is not None:
            return self._cache

        logger.debug("Loading environment variable data")

        values: dict[str, Any] = {}

        # Load from prefix
        if self.config.env_prefix:
            prefix = self.config.env_prefix
            for key, value in os.environ.items():
                if key.startswith(prefix):
                    param_name = key[len(prefix):].lower()
                    values[param_name] = value

        # Load from explicit mapping
        if self.config.env_mapping:
            for env_var, param_name in self.config.env_mapping.items():
                value = os.environ.get(env_var)
                if value is not None:
                    values[param_name] = value

        if not values:
            logger.warning("No environment variables found for data source")
            return []

        # Apply mapping
        values = self._apply_mapping(values)

        # Create single parameter set
        parameter_set = ParameterSet(
            name="env_parameters",
            values=values,
            description="Parameters loaded from environment variables",
        )

        self._cache = [parameter_set]
        logger.info("Loaded environment data", params=list(values.keys()))
        return self._cache

    def validate(self) -> list[str]:
        """Validate environment data source configuration.

        Returns:
            List of validation errors
        """
        errors = []

        if not self.config.env_prefix and not self.config.env_mapping:
            errors.append(
                "Environment data source requires 'env_prefix' or 'env_mapping'"
            )

        # Check if mapped env vars exist
        if self.config.env_mapping:
            missing = [
                var for var in self.config.env_mapping.keys()
                if var not in os.environ
            ]
            if missing:
                errors.append(
                    f"Environment variables not found: {', '.join(missing)}"
                )

        return errors


class DataSourceFactory:
    """Factory for creating data source instances.

    Creates the appropriate data source handler based on configuration.

    Example:
        config = DataSource(type=DataSourceType.CSV, path="data.csv")
        source = DataSourceFactory.create(config)
        parameter_sets = source.load()
    """

    _sources: dict[DataSourceType, type[BaseDataSource]] = {
        DataSourceType.INLINE: InlineDataSource,
        DataSourceType.CSV: CSVDataSource,
        DataSourceType.JSON: JSONDataSource,
        DataSourceType.ENV: EnvironmentDataSource,
    }

    @classmethod
    def create(cls, config: DataSource) -> BaseDataSource:
        """Create a data source instance from configuration.

        Args:
            config: DataSource configuration

        Returns:
            Appropriate BaseDataSource implementation

        Raises:
            DataSourceError: If data source type is not supported
        """
        source_class = cls._sources.get(config.type)

        if source_class is None:
            raise DataSourceError(
                f"Unsupported data source type: {config.type}. "
                f"Supported types: {list(cls._sources.keys())}"
            )

        return source_class(config)

    @classmethod
    def create_from_dict(cls, config_dict: dict[str, Any]) -> BaseDataSource:
        """Create a data source from a dictionary configuration.

        Args:
            config_dict: Dictionary with data source configuration

        Returns:
            Appropriate BaseDataSource implementation
        """
        config = DataSource(**config_dict)
        return cls.create(config)

    @classmethod
    def register(
        cls,
        source_type: DataSourceType,
        source_class: type[BaseDataSource],
    ) -> None:
        """Register a custom data source type.

        Args:
            source_type: Data source type identifier
            source_class: Data source class to register
        """
        cls._sources[source_type] = source_class

    @classmethod
    def get_supported_types(cls) -> list[str]:
        """Get list of supported data source types.

        Returns:
            List of supported type names
        """
        return [t.value for t in cls._sources.keys()]

    @classmethod
    def validate_config(cls, config: DataSource) -> list[str]:
        """Validate a data source configuration without loading.

        Args:
            config: DataSource configuration

        Returns:
            List of validation errors (empty if valid)
        """
        try:
            source = cls.create(config)
            return source.validate()
        except DataSourceError as e:
            return [str(e)]
