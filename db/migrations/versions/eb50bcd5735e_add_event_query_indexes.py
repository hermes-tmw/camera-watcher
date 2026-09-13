"""add event-query performance indexes

Revision ID: eb50bcd5735e
Revises: 4df97ce751c5
Create Date: 2026-09-13 10:13:00

Adds four indexes to accelerate the /watcher/events browser endpoint's
date-range + camera filter (previously full sequential scans on 472k-row
event_observations). These were applied directly to the live DB via
CONCURRENTLY as an emergency fix; this migration makes them durable and
reproducible.

Idempotency: `if_not_exists=True` renders CREATE INDEX IF NOT EXISTS on
PostgreSQL, so this migration is a safe no-op against a DB that already has
the indexes (the live DB does).
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'eb50bcd5735e'
down_revision: Union[str, None] = '4df97ce751c5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        'idx_events_capture_time',
        'event_observations',
        ['capture_time'],
        if_not_exists=True,
    )
    op.create_index(
        'idx_events_camera_capture',
        'event_observations',
        ['camera', 'capture_time'],
        if_not_exists=True,
    )
    op.create_index(
        'idx_labelings_event_id',
        'labelings',
        ['event_id'],
        if_not_exists=True,
    )
    op.create_index(
        'idx_intermediate_results_event_id',
        'intermediate_results',
        ['event_id'],
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index('idx_intermediate_results_event_id', table_name='intermediate_results')
    op.drop_index('idx_labelings_event_id', table_name='labelings')
    op.drop_index('idx_events_camera_capture', table_name='event_observations')
    op.drop_index('idx_events_capture_time', table_name='event_observations')
