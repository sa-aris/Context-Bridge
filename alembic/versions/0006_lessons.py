"""failure memory: lessons learned

Revision ID: 0006_lessons
Revises: 0005_entity_aliases
Create Date: 2024-01-06 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_lessons"
down_revision: str | None = "0005_entity_aliases"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "lessons",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("namespace", sa.String(length=256), nullable=False),
        sa.Column("trigger", sa.Text(), nullable=False),
        sa.Column("guidance", sa.Text(), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("embedding", sa.JSON(), nullable=False),
        sa.Column("times_seen", sa.Integer(), nullable=False),
        sa.Column("times_helpful", sa.Integer(), nullable=False),
        sa.Column("created_by", sa.String(length=128), nullable=True),
        sa.Column("source_session", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_lessons_namespace", "lessons", ["namespace"])


def downgrade() -> None:
    op.drop_index("ix_lessons_namespace", table_name="lessons")
    op.drop_table("lessons")
