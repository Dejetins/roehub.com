import { useEffect, useMemo, useRef, useState } from "react";
import {
  Pulse,
  Bell,
  Brain,
  ChartLineUp,
  CheckCircle,
  CirclesFour,
  Command,
  Database,
  Gear,
  House,
  List,
  MapTrifold,
  MoonStars,
  PlugsConnected,
  SidebarSimple,
  SignOut,
  Sun,
  TestTube,
  Warning,
  X,
} from "@phosphor-icons/react";
import { IconButton, InspectorRow, NavTreeItem, StatusBadge, ThemePicker } from "./components.jsx";
import { notifications, strategies, themes } from "./data.js";
import {
  BacktestsPage,
  ConnectionsPage,
  LivePage,
  LoginPage,
  ModelsPage,
  OverviewPage,
  ProgramMapPage,
  SettingsPage,
  StateGalleryPage,
  StrategiesPage,
} from "./pages.jsx";

const pageConfig = {
  overview: { label: "Overview", icon: House, group: "Operate" },
  strategies: { label: "Strategies", icon: ChartLineUp, group: "Build" },
  backtests: { label: "Backtests", icon: TestTube, group: "Research" },
  live: { label: "Live", icon: Pulse, group: "Operate" },
  models: { label: "Models", icon: Brain, group: "Research" },
  connections: { label: "Connections", icon: PlugsConnected, group: "System" },
  settings: { label: "Settings", icon: Gear, group: "System" },
  map: { label: "Program map", icon: MapTrifold, group: "System" },
  states: { label: "UI states", icon: CirclesFour, group: "System" },
};

const navGroups = [
  { label: "Operate", pages: ["overview", "live"] },
  { label: "Build", pages: ["strategies"] },
  { label: "Research", pages: ["backtests", "models"] },
  { label: "System", pages: ["connections", "settings", "map", "states"] },
];

const commandActions = [
  { label: "Go to portfolio overview", hint: "G O", page: "overview" },
  { label: "Open strategy library", hint: "G S", page: "strategies" },
  { label: "Configure a backtest", hint: "G B", page: "backtests" },
  { label: "Inspect live operations", hint: "G L", page: "live" },
  { label: "Open model registry", hint: "G M", page: "models" },
  { label: "View the complete program map", hint: "G P", page: "map" },
];

function readInitialPage() {
  const hash = window.location.hash.replace("#", "");
  return hash === "login" || pageConfig[hash] ? hash : "overview";
}

function ContextNavigation({ page, onNavigate, collapsed, onToggle, onCommand, onNotifications, onTheme, onUser, notificationCount }) {
  return (
    <nav className={`context-navigation ${collapsed ? "is-collapsed" : ""}`} aria-label="Primary navigation">
      <div className="context-heading">
        <button className="context-brand" aria-label="Roehub overview" onClick={() => onNavigate("overview")}>
          <ChartLineUp size={20} weight="bold" />
          <span className="context-brand-copy"><span>Roehub workspace</span><strong>Quant operations</strong></span>
        </button>
        <IconButton label={collapsed ? "Expand navigation" : "Collapse navigation"} onClick={onToggle}><SidebarSimple size={17} /></IconButton>
      </div>
      <div className="context-scroll">
        {navGroups.map((group) => (
          <section key={group.label}>
            <h2>{group.label}</h2>
            {group.pages.map((id) => {
              const item = pageConfig[id];
              const Icon = item.icon;
              return <NavTreeItem key={id} icon={<Icon size={17} />} label={item.label} active={page === id} count={id === "live" ? 2 : id === "backtests" ? 3 : undefined} onClick={() => onNavigate(id)} />;
            })}
          </section>
        ))}
      </div>
      <div className="context-tools" aria-label="Workspace tools">
        <button className="nav-tree-item" aria-label="Command search" onClick={onCommand}><Command size={17} /><span>Command search</span></button>
        <button className="nav-tree-item" aria-label="Notifications" onClick={onNotifications}><Bell size={17} /><span>Notifications</span>{notificationCount > 0 && <small>{notificationCount}</small>}</button>
        <button className="nav-tree-item" aria-label="Theme" onClick={onTheme}><MoonStars size={17} /><span>Theme</span></button>
      </div>
      <button className="context-footer" aria-label="User menu" onClick={onUser}><span className="user-avatar">UD</span><span><strong>Ultra</strong><small>Capabilities from API</small></span></button>
    </nav>
  );
}

function DocumentTabs({ tabs, page, onNavigate, onClose }) {
  return (
    <div className="document-tabs" role="tablist" aria-label="Open workspaces">
      {tabs.map((id) => { const config = pageConfig[id]; const Icon = config.icon; return (
        <div key={id} className={page === id ? "is-active" : ""}>
          <button role="tab" aria-selected={page === id} onClick={() => onNavigate(id)}><Icon size={14} />{config.label}</button>
          {tabs.length > 1 && <IconButton label={`Close ${config.label}`} onClick={() => onClose(id)}><X size={12} /></IconButton>}
        </div>
      ); })}
    </div>
  );
}

function Inspector({ page, selectedStrategy, onClose, mobileOpen, onNavigate }) {
  const pageInfo = pageConfig[page];
  return (
    <aside className={`inspector ${mobileOpen ? "is-mobile-open" : ""}`} aria-label="Inspector">
      <header><div><span>Inspector</span><strong>{selectedStrategy?.name || pageInfo.label}</strong></div><IconButton label="Close inspector" onClick={onClose}><X size={16} /></IconButton></header>
      <div className="inspector-scroll">
        {selectedStrategy ? (
          <>
            <section><h2>Selection</h2><InspectorRow label="State" value={selectedStrategy.state} tone={selectedStrategy.state === "Running" ? "success" : "danger"} /><InspectorRow label="Venue" value={selectedStrategy.venue} /><InspectorRow label="Symbols" value={selectedStrategy.symbols} /><InspectorRow label="Model" value={selectedStrategy.model} /></section>
            <section><h2>Performance</h2><InspectorRow label="P&L" value={selectedStrategy.pnl} tone={selectedStrategy.pnl.startsWith("-") ? "danger" : "success"} /><InspectorRow label="Return" value={selectedStrategy.pnlPct} /><InspectorRow label="Drawdown" value={selectedStrategy.drawdown} /><InspectorRow label="Updated" value={selectedStrategy.updated} /></section>
            <section><h2>Readiness</h2><InspectorRow label="Data feed" value="Healthy" tone="success" /><InspectorRow label="Model" value="Loaded" tone="success" /><InspectorRow label="Risk limits" value="Healthy" tone="success" /><InspectorRow label="Execution" value="Reconciled" tone="success" /><InspectorRow label="Alerts" value="1 warning" tone="warning" /></section>
            <section><h2>Links</h2><InspectorRow label="Backtest" value="BT-8421" /><InspectorRow label="Dataset" value="Crypto Uni v3" /></section>
            <button className="button button-primary inspector-action" onClick={() => onNavigate("strategies")}>Open workspace</button>
          </>
        ) : (
          <>
            <section><h2>Workspace</h2><InspectorRow label="Area" value={pageInfo.group} /><InspectorRow label="View" value={pageInfo.label} /><InspectorRow label="Access" value="Resolved by backend" /><InspectorRow label="Context" value="No object selected" /></section>
            <section><h2>Prototype boundary</h2><InspectorRow label="Environment" value="Illustrative Testnet" /><InspectorRow label="Data" value="Local fixtures" /><InspectorRow label="API" value="Not connected" /><InspectorRow label="Mutations" value="Demonstration only" /></section>
            <section><h2>Shortcuts</h2><div className="shortcut-list"><span><kbd>⌘</kbd><kbd>K</kbd>Command search</span><span><kbd>G</kbd><kbd>B</kbd>Backtests</span><span><kbd>G</kbd><kbd>L</kbd>Live operations</span></div></section>
          </>
        )}
      </div>
    </aside>
  );
}

function useDialogFocus() {
  const dialogRef = useRef(null);
  const previousFocusRef = useRef(null);
  useEffect(() => {
    previousFocusRef.current = document.activeElement;
    const focusable = dialogRef.current?.querySelector('input, button, [href], [tabindex]:not([tabindex="-1"])');
    focusable?.focus();
    return () => previousFocusRef.current?.focus?.();
  }, []);
  const onKeyDown = (event) => {
    if (event.key !== "Tab") return;
    const focusable = [...(dialogRef.current?.querySelectorAll('input, button:not([disabled]), [href], [tabindex]:not([tabindex="-1"])') || [])];
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable.at(-1);
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
    if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
  };
  return { dialogRef, onKeyDown };
}

function CommandPalette({ onClose, onNavigate }) {
  const [query, setQuery] = useState("");
  const { dialogRef, onKeyDown } = useDialogFocus();
  const results = commandActions.filter((action) => action.label.toLowerCase().includes(query.toLowerCase()));
  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <section ref={dialogRef} className="command-palette" role="dialog" aria-modal="true" aria-label="Command search" onKeyDown={onKeyDown} onMouseDown={(event) => event.stopPropagation()}>
        <label><Command size={19} /><span className="sr-only">Search commands</span><input autoFocus value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search workspaces and actions…" /><kbd>ESC</kbd></label>
        <div>{results.map((action) => <button key={action.page} onClick={() => { onNavigate(action.page); onClose(); }}><span>{action.label}</span><kbd>{action.hint}</kbd></button>)}{results.length === 0 && <p>No matching command</p>}</div>
        <footer><span><kbd>↑</kbd><kbd>↓</kbd>Navigate</span><span><kbd>↵</kbd>Open</span></footer>
      </section>
    </div>
  );
}

function NotificationCenter({ onClose, onNavigate, unreadCount, onMarkAll }) {
  const { dialogRef, onKeyDown } = useDialogFocus();
  return (
    <aside ref={dialogRef} className="overlay-panel notification-center" role="dialog" aria-modal="true" aria-label="Notifications" onKeyDown={onKeyDown}>
      <header><div><strong>Notifications</strong><span>{unreadCount} unread · all workspaces</span></div><IconButton label="Close notifications" onClick={onClose}><X size={16} /></IconButton></header>
      <div className="notification-list">{notifications.map((item) => <button key={item.id} className={unreadCount > 0 && item.unread ? "is-unread" : ""} onClick={() => { onNavigate(item.title.includes("Backtest") ? "backtests" : item.title.includes("stream") ? "connections" : "live"); onClose(); }}><i className={`notification-level level-${item.level}`}>{item.level === "warning" || item.level === "danger" ? <Warning size={15} weight="fill" /> : <CheckCircle size={15} weight="fill" />}</i><span><strong>{item.title}</strong><small>{item.detail}</small></span><time>{item.time}</time></button>)}</div>
      <footer><button className="button button-secondary" disabled={unreadCount === 0} onClick={onMarkAll}>Mark all as read</button><button className="text-button" onClick={() => { onNavigate("settings"); onClose(); }}>Notification settings</button></footer>
    </aside>
  );
}

function UserMenu({ onClose, onLogout, onNavigate }) {
  return (
    <div className="popover user-popover" role="dialog" aria-label="User menu">
      <div className="user-card"><span className="user-avatar">UD</span><span><strong>Ultra</strong><small>Capabilities resolved by API · Testnet</small></span></div>
      <button onClick={() => { onNavigate("settings"); onClose(); }}><Gear size={16} />Workspace settings</button>
      <button onClick={onLogout}><SignOut size={16} />Sign out</button>
    </div>
  );
}

function ActivityDrawer({ open, onToggle }) {
  return (
    <section className={`activity-drawer ${open ? "is-open" : ""}`}>
      <button className="activity-handle" onClick={onToggle}><Pulse size={15} /><strong>Activity</strong><StatusBadge tone="success">4 streams</StatusBadge><span>{open ? "Hide" : "Show"}</span></button>
      {open && <div className="activity-content"><div><time>12:27:51</time><CheckCircle size={14} weight="fill" /><span>BTCUSDT order filled</span><small>Momentum Breakout v7</small></div><div><time>12:27:42</time><Database size={14} /><span>Model checkpoint saved</span><small>RL Alpha v15-rc2</small></div><div><time>12:27:20</time><Warning size={14} weight="fill" /><span>Drawdown soft limit approaching</span><small>Momentum Breakout v7</small></div></div>}
    </section>
  );
}

function MobileBottomNavigation({ page, onNavigate, onMore }) {
  return (
    <nav className="mobile-bottom-nav" aria-label="Mobile navigation">
      {["overview", "strategies", "backtests", "live"].map((id) => { const Icon = pageConfig[id].icon; return <button key={id} className={page === id ? "is-active" : ""} onClick={() => onNavigate(id)}><Icon size={20} weight={page === id ? "fill" : "regular"} /><span>{pageConfig[id].label}</span></button>; })}
      <button onClick={onMore}><List size={20} /><span>More</span></button>
    </nav>
  );
}

export function App() {
  const [page, setPage] = useState(readInitialPage);
  const [loggedIn, setLoggedIn] = useState(() => readInitialPage() !== "login");
  const [theme, setTheme] = useState(() => localStorage.getItem("roehub-theme") || "graphite");
  const [tabs, setTabs] = useState(() => {
    const initial = readInitialPage();
    const base = ["overview", "models", "backtests"];
    return initial === "login" ? base : [initial, ...base.filter((tab) => tab !== initial)];
  });
  const [navCollapsed, setNavCollapsed] = useState(true);
  const [mobileNav, setMobileNav] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(() => window.innerWidth > 820);
  const [activityOpen, setActivityOpen] = useState(false);
  const [commandOpen, setCommandOpen] = useState(false);
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  const [notificationCount, setNotificationCount] = useState(2);
  const [themeOpen, setThemeOpen] = useState(false);
  const [userOpen, setUserOpen] = useState(false);
  const [selectedStrategy, setSelectedStrategy] = useState(strategies[2]);
  const [toast, setToast] = useState("");

  const navigate = (next) => {
    if (!pageConfig[next]) return;
    setPage(next);
    setTabs((current) => [next, ...current.filter((tab) => tab !== next)].slice(0, 5));
    setSelectedStrategy(null);
    setMobileNav(false);
    window.history.pushState({}, "", `#${next}`);
  };

  useEffect(() => {
    const onPop = () => { const next = readInitialPage(); if (next !== "login") { setPage(next); setSelectedStrategy(null); setLoggedIn(true); } };
    const onKey = (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") { event.preventDefault(); setCommandOpen(true); }
      if (event.key === "Escape") { setCommandOpen(false); setNotificationsOpen(false); setThemeOpen(false); setUserOpen(false); setMobileNav(false); }
    };
    window.addEventListener("popstate", onPop);
    window.addEventListener("keydown", onKey);
    return () => { window.removeEventListener("popstate", onPop); window.removeEventListener("keydown", onKey); };
  }, []);

  useEffect(() => { localStorage.setItem("roehub-theme", theme); }, [theme]);
  useEffect(() => {
    const closeInspectorOnSmallScreens = () => { if (window.innerWidth <= 820) setInspectorOpen(false); };
    window.addEventListener("resize", closeInspectorOnSmallScreens);
    return () => window.removeEventListener("resize", closeInspectorOnSmallScreens);
  }, []);
  useEffect(() => { if (!toast) return undefined; const timer = window.setTimeout(() => setToast(""), 2600); return () => window.clearTimeout(timer); }, [toast]);

  const closeTab = (id) => {
    setTabs((current) => {
      const next = current.filter((tab) => tab !== id);
      if (page === id) navigate(next.at(-1) || "overview");
      return next.length ? next : ["overview"];
    });
  };

  const openStrategy = (strategy) => { setSelectedStrategy(strategy); setInspectorOpen(true); };
  const renderPage = useMemo(() => {
    const common = { onToast: setToast };
    if (page === "overview") return <OverviewPage onNavigate={navigate} onSelectStrategy={openStrategy} />;
    if (page === "strategies") return <StrategiesPage onNavigate={navigate} onSelectStrategy={openStrategy} {...common} />;
    if (page === "backtests") return <BacktestsPage {...common} />;
    if (page === "live") return <LivePage onSelectStrategy={openStrategy} {...common} />;
    if (page === "models") return <ModelsPage />;
    if (page === "connections") return <ConnectionsPage {...common} />;
    if (page === "settings") return <SettingsPage activeTheme={theme} onThemeChange={setTheme} />;
    if (page === "map") return <ProgramMapPage onNavigate={navigate} />;
    return <StateGalleryPage />;
  }, [page, theme]);

  if (!loggedIn) return <LoginPage activeTheme={theme} onThemeChange={setTheme} onLogin={() => { setLoggedIn(true); navigate("overview"); }} />;

  return (
    <div className="app-shell" data-theme={theme} data-nav-collapsed={navCollapsed} data-inspector-open={inspectorOpen}>
      <ContextNavigation page={page} onNavigate={navigate} collapsed={navCollapsed} onToggle={() => setNavCollapsed(!navCollapsed)} onCommand={() => setCommandOpen(true)} onNotifications={() => setNotificationsOpen(!notificationsOpen)} onTheme={() => setThemeOpen(!themeOpen)} onUser={() => setUserOpen(!userOpen)} notificationCount={notificationCount} />
      <header className="top-toolbar">
        <div className="mobile-toolbar-left"><IconButton label="Open navigation" onClick={() => setMobileNav(true)}><List size={19} /></IconButton><ChartLineUp size={19} weight="bold" /></div>
        <div className="breadcrumb"><span>Roehub</span><b>/</b><strong>{pageConfig[page].group}</strong><b>/</b><span>{pageConfig[page].label}</span></div>
        <button className="command-trigger" onClick={() => setCommandOpen(true)}><Command size={16} /><span>Search or run a command</span><kbd>⌘ K</kbd></button>
        <div className="toolbar-actions"><StatusBadge tone="success">Testnet</StatusBadge><IconButton label="Toggle inspector" onClick={() => setInspectorOpen(!inspectorOpen)}><SidebarSimple size={18} /></IconButton><button className="user-chip" onClick={() => setUserOpen(!userOpen)}><span className="user-avatar">UD</span><span>Ultra</span></button></div>
      </header>
      <DocumentTabs tabs={tabs} page={page} onNavigate={navigate} onClose={closeTab} />
      <main className="app-content">{renderPage}</main>
      <Inspector page={page} selectedStrategy={selectedStrategy} onClose={() => setInspectorOpen(false)} mobileOpen={inspectorOpen} onNavigate={navigate} />
      <ActivityDrawer open={activityOpen} onToggle={() => setActivityOpen(!activityOpen)} />
      <footer className="status-bar"><span><i className="status-light" />Connected</span><span>API 14 ms</span><span>Compute 6 / 6</span><span>Data <b className="text-warning">1 delayed</b></span><button onClick={() => setActivityOpen(!activityOpen)}>Last event 12:27:51</button><span className="status-spacer" /><span>{themes.find((item) => item.id === theme)?.name}</span><span>v2 concept</span></footer>
      <MobileBottomNavigation page={page} onNavigate={navigate} onMore={() => setMobileNav(true)} />

      {mobileNav && <div className="mobile-nav-backdrop" onMouseDown={() => setMobileNav(false)}><div className="mobile-nav-sheet" onMouseDown={(event) => event.stopPropagation()}><header><div className="login-logo"><ChartLineUp size={20} weight="bold" /><strong>Roehub</strong></div><IconButton label="Close navigation" onClick={() => setMobileNav(false)}><X size={17} /></IconButton></header>{navGroups.map((group) => <section key={group.label}><h2>{group.label}</h2>{group.pages.map((id) => { const Icon = pageConfig[id].icon; return <NavTreeItem key={id} icon={<Icon size={17} />} label={pageConfig[id].label} active={page === id} onClick={() => navigate(id)} />; })}</section>)}</div></div>}
      {commandOpen && <CommandPalette onClose={() => setCommandOpen(false)} onNavigate={navigate} />}
      {notificationsOpen && <NotificationCenter onClose={() => setNotificationsOpen(false)} onNavigate={navigate} unreadCount={notificationCount} onMarkAll={() => setNotificationCount(0)} />}
      {themeOpen && <div className="anchored-popover theme-anchor"><ThemePicker activeTheme={theme} onChange={setTheme} onClose={() => setThemeOpen(false)} /></div>}
      {userOpen && <div className="anchored-popover user-anchor"><UserMenu onClose={() => setUserOpen(false)} onNavigate={navigate} onLogout={() => { setLoggedIn(false); setUserOpen(false); window.history.pushState({}, "", "#login"); }} /></div>}
      {toast && <div className="toast" role="status"><CheckCircle size={18} weight="fill" />{toast}</div>}
    </div>
  );
}
