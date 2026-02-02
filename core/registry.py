"""Validator registry for pluggable validation system."""

from validators.base import BaseValidator


class ValidatorRegistry:
    """Registry for managing available validators."""

    def __init__(self):
        self._validators: dict[str, BaseValidator] = {}

    def register(self, validator: BaseValidator) -> None:
        """Register a validator.

        Args:
            validator: The validator instance to register
        """
        self._validators[validator.name] = validator

    def get_enabled_validators(self, enabled: set[str] | None = None) -> list[BaseValidator]:
        """Get list of enabled validators.

        Args:
            enabled: Set of validator names to enable, or None for defaults

        Returns:
            List of enabled validator instances
        """
        if enabled is None:
            # Use default enabled validators
            return [v for v in self._validators.values() if v.default_enabled]
        else:
            # Use explicitly enabled validators
            return [v for name, v in self._validators.items() if name in enabled]

    def list_all(self) -> dict[str, dict]:
        """List all registered validators with metadata.

        Returns:
            Dict mapping validator names to their metadata
        """
        return {
            name: {
                "name": validator.name,
                "description": validator.description,
                "default_enabled": validator.default_enabled,
            }
            for name, validator in self._validators.items()
        }


# Singleton registry instance
_registry = ValidatorRegistry()


def get_registry() -> ValidatorRegistry:
    """Get the global validator registry."""
    return _registry
