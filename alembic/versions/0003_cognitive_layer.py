"""cognitive layer: feedback, conflicts, knowledge graph

Revision ID: 0003_cognitive_layer
Revises: 0002_parent_documents
Create Date: 2024-01-03 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_cognitive_layer"
down_revision: str | None = "0002_parent_documents"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "feedback",
        sa.Column("memory_id", sa.String(length=64), nullable=False),
        sa.Column("namespace", sa.String(length=256), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("votes", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("memory_id"),
    )
    op.create_table(
        "conflicts",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("namespace", sa.String(length=256), nullable=False),
        sa.Column("memory_id_a", sa.String(length=64), nullable=False),
        sa.Column("memory_id_b", sa.String(length=64), nullable=False),
        sa.Column("similarity", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("winner_id", sa.String(length=64), nullable=True),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_conflicts_namespace_status", "conflicts", ["namespace", "status"])
    op.create_table(
        "graph_nodes",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("namespace", sa.String(length=256), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("namespace", "name", name="uq_graph_node"),
    )
    op.create_table(
        "graph_edges",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("namespace", sa.String(length=256), nullable=False),
        sa.Column("source", sa.String(length=256), nullable=False),
        sa.Column("relation", sa.String(length=128), nullable=False),
        sa.Column("target", sa.String(length=256), nullable=False),
        sa.Column("memory_id", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_graph_edges_ns_source", "graph_edges", ["namespace", "source"])


def downgrade() -> None:
    op.drop_index("ix_graph_edges_ns_source", table_name="graph_edges")
    op.drop_table("graph_edges")
    op.drop_table("graph_nodes")
    op.drop_index("ix_conflicts_namespace_status", table_name="conflicts")
    op.drop_table("conflicts")
    op.drop_table("feedback")
