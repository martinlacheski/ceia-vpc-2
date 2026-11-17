"""Servicio del medidor: agua o energía (decide qué modelos leen sus fotos).

Los medidores existentes quedan como agua, que era el único servicio.

Revision ID: 0003_meter_service
Revises: 0002_reading_review
Create Date: 2026-11-17 18:00:00
"""
from alembic import op
import sqlalchemy as sa


revision = '0003_meter_service'
down_revision = '0002_reading_review'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('meters') as batch:
        batch.add_column(sa.Column('service', sa.String(length=10), nullable=False, server_default='water'))


def downgrade() -> None:
    with op.batch_alter_table('meters') as batch:
        batch.drop_column('service')
