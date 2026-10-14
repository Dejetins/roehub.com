"""Seed one invited local identity in the disposable lifecycle verification database.

This is storage/isolation fixture setup, not authentication proof. Local passkey
login and recovery are exercised separately by verify_greenfield_lifecycle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from uuid import uuid4

import psycopg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--email', required=True)
    parser.add_argument('--fixture-id', choices=('a', 'b'), required=True)
    args = parser.parse_args()
    user_id = uuid4()
    email_hash = hashlib.sha256(args.email.strip().lower().encode()).hexdigest()
    with psycopg.connect(os.environ['IDENTITY_PG_DSN']) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT display_name FROM identity_installations")
            if cursor.fetchall() != [('Stage 23 Disposable',)]:
                raise RuntimeError('local fixture requires the disposable lifecycle installation')
            cursor.execute(
                "SELECT invitation_id, organization_id, role FROM identity_invitations "
                "WHERE recipient_email_sha256=%s AND status='pending' "
                "AND expires_at > now() FOR UPDATE", (email_hash,),
            )
            rows = cursor.fetchall()
            if len(rows) != 1:
                raise RuntimeError('fixture requires exactly one pending invitation')
            invitation_id, organization_id, role = rows[0]
            cursor.execute(
                "INSERT INTO identity_users (user_id, paid_level, created_at, is_deleted) "
                "VALUES (%s, 'free', now(), false)", (user_id,),
            )
            cursor.execute(
                "INSERT INTO identity_memberships "
                "(organization_id,user_id,role,status,created_at,updated_at) "
                "VALUES (%s,%s,%s,'active',now(),now())", (organization_id,user_id,role),
            )
            cursor.execute(
                "UPDATE identity_invitations SET status='accepted', "
                "accepted_by_user_id=%s, accepted_at=now() WHERE invitation_id=%s",
                (user_id,invitation_id),
            )
    print(json.dumps({'accepted_invitation_count':1, 'provisioned':True,
                      'fixture_id':args.fixture_id, 'user_id':str(user_id)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
