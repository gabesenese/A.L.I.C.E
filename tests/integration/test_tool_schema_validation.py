import pytest

from ai.contracts import (
    ToolSchemaValidationError,
    validate_tool_invocation_payload,
    validate_tool_result_payload,
)


def test_validate_tool_invocation_payload_accepts_valid_shape():
    validate_tool_invocation_payload(
        {
            "tool_name": "notes",
            "action": "notes:search",
            "params": {"query": "project"},
        }
    )


def test_validate_tool_invocation_payload_rejects_missing_action():
    with pytest.raises(ToolSchemaValidationError):
        validate_tool_invocation_payload({"tool_name": "notes", "params": {}})


def test_validate_tool_result_payload_accepts_valid_shape():
    validate_tool_result_payload(
        {
            "success": True,
            "tool_name": "notes",
            "action": "notes:search",
            "data": {"response": "ok"},
            "diagnostics": {"source": "test"},
        }
    )


def test_validate_tool_result_payload_rejects_invalid_success_type():
    with pytest.raises(ToolSchemaValidationError):
        validate_tool_result_payload(
            {
                "success": "yes",
                "tool_name": "notes",
                "action": "notes:search",
                "data": {},
            }
        )
