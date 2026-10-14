import type {ReactNode} from 'react';

const actionOrder = ['create', 'search', 'refresh', 'filter', 'edit'] as const;
export type LibraryActions = Partial<Record<typeof actionOrder[number], ReactNode>>;

/** Shared title and action rows for the workspace's left collection panels. */
export function LibraryHeader({title, headingId, count, actions, compactActions = false}: {
  title: ReactNode;
  headingId?: string;
  count?: ReactNode;
  actions?: LibraryActions;
  compactActions?: boolean;
}) {
  const visibleActions = actionOrder.filter(action => actions?.[action]);
  return <header className="panel-head library-head">
    <div className="library-title"><h2 id={headingId}>{title}</h2>{count}</div>
    {actions && <div className="library-actions">{visibleActions.map((action, index) =>
      <div key={action} className="library-action" data-action={action}
        style={{gridColumn: (compactActions ? index : actionOrder.indexOf(action)) + 1}}>{actions[action]}</div>
    )}</div>}
  </header>;
}
