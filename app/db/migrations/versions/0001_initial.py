"""Esquema inicial: usuarios, rutas, medidores, análisis y lecturas.

Revision ID: 0001_initial
Revises: 
Create Date: 2026-11-14 11:33:16.149322
"""
from alembic import op
import sqlalchemy as sa


revision = '0001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('users',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('username', sa.String(length=50), nullable=False),
    sa.Column('full_name', sa.String(length=100), nullable=False),
    sa.Column('role', sa.String(length=10), nullable=False),
    sa.Column('password_hash', sa.String(length=200), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('username')
    )
    op.create_table('routes',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=50), nullable=False),
    sa.Column('assignee_id', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['assignee_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name')
    )
    op.create_table('meters',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('code', sa.String(length=50), nullable=False),
    sa.Column('address', sa.String(length=200), nullable=False),
    sa.Column('lat', sa.Float(), nullable=False),
    sa.Column('lon', sa.Float(), nullable=False),
    sa.Column('route_id', sa.Integer(), nullable=False),
    sa.Column('sample_photo', sa.String(length=200), nullable=True),
    sa.ForeignKeyConstraint(['route_id'], ['routes.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('code')
    )
    op.create_table('analyses',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('meter_id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('photo_path', sa.String(length=200), nullable=False),
    sa.Column('proposed_value', sa.String(length=50), nullable=True),
    sa.Column('confidence', sa.Float(), nullable=True),
    sa.Column('used', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['meter_id'], ['meters.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('readings',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('meter_id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('analysis_id', sa.String(length=36), nullable=False),
    sa.Column('value', sa.String(length=50), nullable=False),
    sa.Column('proposed_value', sa.String(length=50), nullable=True),
    sa.Column('corrected', sa.Boolean(), nullable=False),
    sa.Column('confidence', sa.Float(), nullable=True),
    sa.Column('lat', sa.Float(), nullable=True),
    sa.Column('lon', sa.Float(), nullable=True),
    sa.Column('accuracy_m', sa.Float(), nullable=True),
    sa.Column('taken_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['analysis_id'], ['analyses.id'], ),
    sa.ForeignKeyConstraint(['meter_id'], ['meters.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('analysis_id')
    )
    with op.batch_alter_table('readings', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_readings_meter_id'), ['meter_id'], unique=False)



def downgrade() -> None:
    with op.batch_alter_table('readings', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_readings_meter_id'))

    op.drop_table('readings')
    op.drop_table('analyses')
    op.drop_table('meters')
    op.drop_table('routes')
    op.drop_table('users')
