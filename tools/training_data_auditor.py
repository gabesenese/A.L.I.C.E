"""Compatibility wrapper for the canonical auditor module."""

from tools.auditing.training_data_auditor import (
    LearningDataQAAuditor,
    TrainingDataAuditor,
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
