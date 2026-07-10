#!/bin/sh
set -e

# Apply database migrations when Postgres is available.
if [ -n "${DATABASE_URL:-}" ]; then
  echo "Running alembic migrations..."
  alembic upgrade head
fi

exec "$@"
