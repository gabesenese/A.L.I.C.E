"""Compatibility wrapper for the canonical auditor module."""

from tools.auditing.training_data_auditor import (
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
