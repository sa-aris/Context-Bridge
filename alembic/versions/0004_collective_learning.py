"""collective learning: agent profiles and procedures

Revision ID: 0004_collective_learning
Revises: 0003_cognitive_layer
Create Date: 2024-01-04 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_collective_learning"
down_revision: str | None = "0003_cognitive_layer"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_profiles",
        sa.Column("id", sa.String(length=320), nullable=False),
        sa.Column("namespace", sa.String(length=256), nullable=False),
        sa.Column("agent_id", sa.String(length=128), nullable=False),
        sa.Column("writes", sa.Integer(), nullable=False),
        sa.Column("useful", sa.Integer(), nullable=False),
        sa.Column("unhelpful", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_agent_profiles_ns_score", "agent_profiles", ["namespace", "score"])
    op.create_table(
        "procedures",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("namespace", sa.String(length=256), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("steps", sa.JSON(), nullable=False),
        sa.Column("tags", sa.JSON(), nullable=False),
        sa.Column("success_count", sa.Integer(), nullable=False),
        sa.Column("fail_count", sa.Integer(), nullable=False),
        sa.Column("created_by", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_procedures_namespace", "procedures", ["namespace"])


def downgrade() -> None:
    op.drop_index("ix_procedures_namespace", table_name="procedures")
    op.drop_table("procedures")
    op.drop_index("ix_agent_profiles_ns_score", table_name="agent_profiles")
    op.drop_table("agent_profiles")
