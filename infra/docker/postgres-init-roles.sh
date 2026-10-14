#!/bin/sh
set -eu
# Runs only during creation of a fresh embedded cluster. No passwords in argv/logs.
app_password=$(cat /run/roehub-secrets/postgresql-app-password)
monitor_password=$(cat /run/roehub-pg-monitor/monitor-password)
maintenance_password=$(cat /run/roehub-pg-maintenance/password)
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<SQL
CREATE ROLE roehub_app LOGIN PASSWORD '$app_password' NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION;
CREATE ROLE roehub_monitor LOGIN PASSWORD '$monitor_password' NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION;
CREATE ROLE roehub_maintenance LOGIN PASSWORD '$maintenance_password' NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION;
GRANT pg_monitor, pg_use_reserved_connections TO roehub_monitor;
GRANT pg_use_reserved_connections TO roehub_maintenance;
GRANT EXECUTE ON FUNCTION pg_catalog.pg_control_system() TO roehub_app, roehub_monitor, roehub_maintenance;
SQL
