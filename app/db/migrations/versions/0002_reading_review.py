"""Revisión del administrador sobre una lectura (valor, quién y cuándo).

Solo agrega columnas opcionales: las lecturas existentes quedan sin revisión.

Revision ID: 0002_reading_review
Revises: 0001_initial
Create Date: 2026-11-14 13:30:00
"""
from alembic import op
import sqlalchemy as sa


revision = '0002_reading_review'
down_revision = '0001_initial'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # batch_alter_table: SQLite (tests) no permite agregar claves foráneas con ALTER TABLE
    with op.batch_alter_table('readings') as batch:
        batch.add_column(sa.Column('reviewed_value', sa.String(length=50), nullable=True))
        batch.add_column(sa.Column('reviewed_by_id', sa.Integer(), nullable=True))
        batch.add_column(sa.Column('reviewed_at', sa.DateTime(timezone=True), nullable=True))
        batch.create_foreign_key('fk_readings_reviewed_by_id_users', 'users', ['reviewed_by_id'], ['id'])


def downgrade() -> None:
    with op.batch_alter_table('readings') as batch:
        batch.drop_constraint('fk_readings_reviewed_by_id_users', type_='foreignkey')
        batch.drop_column('reviewed_at')
        batch.drop_column('reviewed_by_id')
        batch.drop_column('reviewed_value')
