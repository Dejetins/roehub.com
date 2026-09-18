"""Bounded owned execution facts for the strategy workspace; no mutation methods."""


class PostgresStrategyOperationReader:
    def __init__(self, gateway):
        self.gateway = gateway

    def read(self, *, organization_id, owner_user_id, strategy_id, run_id, paper):
        params = {
            "org": organization_id.value,
            "owner": owner_user_id.value,
            "strategy": strategy_id,
            "run": run_id,
        }
        if paper:
            query = """
                SELECT f.paper_fill_id AS fill_id, f.filled_at AS time, f.fill_price AS price,
                       f.quantity, f.side, f.fee_amount AS fee, f.fee_asset,
                       e.source_type, e.source_ref_json->>'exit_reason' AS exit_reason,
                       COALESCE(s.reference_price, o.reference_price) AS reference_price,
                       s.reason_code AS signal_reason,
                       (SELECT a.reserved_budget FROM strategy_paper_accounting a
                        WHERE a.organization_id=f.organization_id
                          AND a.owner_user_id=f.owner_user_id
                          AND a.strategy_id=f.strategy_id AND a.strategy_run_id=f.strategy_run_id
                        ORDER BY a.created_at,a.accounting_id LIMIT 1) AS initial_cash
                FROM paper_fills f
                JOIN paper_orders o ON o.organization_id=f.organization_id
                  AND o.owner_user_id=f.owner_user_id AND o.paper_order_id=f.paper_order_id
                LEFT JOIN execution_source_events e ON e.organization_id=o.organization_id
                  AND e.owner_user_id=o.owner_user_id AND e.source_event_id=o.source_event_id
                LEFT JOIN strategy_signals s ON s.organization_id=o.organization_id
                  AND s.owner_user_id=o.owner_user_id AND s.signal_id=o.source_signal_id
                WHERE f.organization_id=%(org)s AND f.owner_user_id=%(owner)s
                  AND f.strategy_id=%(strategy)s AND f.strategy_run_id=%(run)s
                ORDER BY f.filled_at,f.paper_fill_id LIMIT 5001
            """
        else:
            query = """
                SELECT f.fill_id, f.filled_at AS time, f.price, f.quantity, o.side,
                       f.fee_amount AS fee, f.fee_asset, e.source_type,
                       e.source_ref_json->>'exit_reason' AS exit_reason,
                       s.reference_price AS reference_price,
                       s.reason_code AS signal_reason
                FROM execution_fills f
                JOIN execution_orders o ON o.organization_id=f.organization_id
                  AND o.owner_user_id=f.owner_user_id AND o.order_id=f.order_id
                JOIN execution_intents i ON i.organization_id=f.organization_id
                  AND i.owner_user_id=f.owner_user_id AND i.intent_id=f.intent_id
                JOIN execution_source_events e ON e.organization_id=i.organization_id
                  AND e.owner_user_id=i.owner_user_id AND e.source_event_id=i.source_event_id
                LEFT JOIN strategy_signals s ON s.organization_id=e.organization_id
                  AND s.owner_user_id=e.owner_user_id AND s.signal_id=e.strategy_signal_id
                WHERE f.organization_id=%(org)s AND f.owner_user_id=%(owner)s
                  AND e.source_ref_json->>'strategy_id'=%(strategy)s::text
                  AND e.source_ref_json->>'strategy_run_id'=%(run)s::text
                ORDER BY f.filled_at,f.fill_id LIMIT 5001
            """
        return self.gateway.fetch_all(query=query, parameters=params)
