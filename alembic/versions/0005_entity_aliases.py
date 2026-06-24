"""ontology alignment: entity aliases

Revision ID: 0005_entity_aliases
Revises: 0004_collective_learning
Create Date: 2024-01-05 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_entity_aliases"
down_revision: str | None = "0004_collective_learning"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "entity_aliases",
        sa.Column("id", sa.String(length=320), nullable=False),
        sa.Column("namespace", sa.String(length=256), nullable=False),
        sa.Column("alias", sa.String(length=256), nullable=False),
        sa.Column("canonical", sa.String(length=256), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_entity_aliases_namespace", "entity_aliases", ["namespace"])


def downgrade() -> None:
    op.drop_index("ix_entity_aliases_namespace", table_name="entity_aliases")
    op.drop_table("entity_aliases")
