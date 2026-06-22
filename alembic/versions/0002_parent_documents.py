"""parent document store for small-to-big retrieval

Revision ID: 0002_parent_documents
Revises: 0001_initial
Create Date: 2024-01-02 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_parent_documents"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "parent_documents",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("namespace", sa.String(length=256), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("parent_documents")
