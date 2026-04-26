from app.logging_config import bind_context, get_logger, set_trace_id


def test_logging_config_fallback_logger_available():
    set_trace_id("trace-123")
    logger = get_logger("alice.test")
    logger = bind_context(logger, component="unit")
    # Should not raise regardless of structlog availability.
    logger.info("fallback logger check")
