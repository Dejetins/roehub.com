import {useEffect, useRef, type ReactNode} from 'react';
import {useBlocker} from 'react-router';
import {useTranslation} from 'react-i18next';
import {RefreshCw} from 'lucide-react';
import {Filter} from 'lucide-react';
import {useLibraryFilterPopup} from './library-filter-popup';
import {ApiError} from './api';
import {LoadingData, ReadStatus} from './loading-data';

export function useWords() {
  const {i18n} = useTranslation();
  const ru = i18n.language.startsWith('ru');
  return (en: string, russian: string) => ru ? russian : en;
}

export function WorkError({error}: {error: unknown}) {
  const w = useWords();
  if (!error) return null;
  const kind = error instanceof ApiError ? error.kind : 'unavailable';
  const messages: Record<string, string> = {
    forbidden: w('Access denied.', 'Нет доступа.'),
    'not-found': w('The resource is unavailable or no longer exists.', 'Ресурс недоступен или больше не существует.'),
    unauthenticated: w('Your session has ended.', 'Сессия завершена.'),
    validation: w('Check the highlighted fields.', 'Проверьте отмеченные поля.'),
    conflict: w('The state has changed. Refresh before continuing.', 'Состояние изменилось. Обновите данные.'),
    'rate-limited': w('Too many requests. Wait before refreshing.', 'Слишком много запросов. Подождите перед обновлением.'),
  };
  const codes:Record<string,string>={
    duplicate_work:w('This instrument already has a download for an overlapping period. Open that request to continue.','Для этого инструмента уже есть загрузка пересекающегося периода. Откройте её, чтобы продолжить.'),
    queue_full:w('The queue has reached 200 unfinished requests. Finish or cancel some before adding more.','В очереди уже 200 незавершённых заданий. Завершите или отмените часть перед добавлением новых.'),
    stale_work_control:w('The download state changed. Refresh it before issuing another command.','Состояние загрузки изменилось. Обновите его перед следующей командой.'),
    work_not_retryable:w('This request cannot be continued: it is not failed or has reached its attempt limit.','Продолжить нельзя: задание не в состоянии ошибки или исчерпан лимит попыток.'),
    work_not_cancellable:w('This request has already finished and cannot be cancelled.','Задание уже завершено, отменить его нельзя.'),
    notification_provider_unavailable:w('No notification provider is configured for this organization.','Для этой организации не настроен провайдер уведомлений.'),
    exchange_control_unavailable:w('The private exchange connection service is unavailable. Public market data is separate.','Сервис приватных биржевых подключений недоступен. Публичные рыночные данные доступны отдельно.'),
    recent_auth_required:w('Confirm your identity again before changing this connection.','Подтвердите личность заново перед изменением подключения.'),
    exchange_connection_in_use:w('Active strategies use this connection. Change their bindings before disconnecting.','Это подключение используют активные стратегии. Измените их привязки перед отключением.'),
    exchange_connection_already_exists:w('This connection already exists. Refresh the list.','Такое подключение уже существует. Обновите список.'),
  };
  return <p className="notice error" role="alert">{error instanceof ApiError && error.outcome === 'unknown'
    ? w('The result is unknown. Refresh to check the saved state before trying again.', 'Результат неизвестен. Обновите данные и проверьте сохранённое состояние перед повтором.')
    : (error instanceof ApiError ? codes[error.code??''] : undefined) ?? messages[kind] ?? w('The service could not be read. Try refreshing.', 'Не удалось получить данные сервиса. Повторите обновление.')}</p>;
}

export function ReadFeedback({pending, retained, initial, error}: {pending: boolean; retained?: boolean; initial?: boolean; error?: unknown}) {
  return <><ReadStatus pending={pending && !initial} retained={retained} />{initial && !error && <LoadingData />}<WorkError error={error} /></>;
}

export function RefreshButton({pending, onClick, disabled=false}: {pending: boolean; onClick: () => void; disabled?: boolean}) {
  const w = useWords();
  return <button className="library-icon-action work-icon" type="button" aria-label={w('Refresh', 'Обновить')} title={w('Refresh', 'Обновить')}
    disabled={pending || disabled} onClick={onClick}><RefreshCw aria-hidden="true" /></button>;
}

export function WorkHeading({title, children}: {title: string; children?: ReactNode}) {
  return <header className="navigator-heading"><h1 id="workspace-heading" tabIndex={-1}>{title}</h1>{children}</header>;
}

export function Field({name, label, error, children}: {name: string; label: string; error?: string; children: ReactNode}) {
  return <div className="field work-field"><label htmlFor={name}>{label}</label>{children}{error && <span id={`${name}-error`} className="work-field-error">{error}</span>}</div>;
}

/** Browser confirmation covers document departures; the router covers history and SPA links. */
export function DirtyGuard({dirty}: {dirty: boolean}) {
  const w = useWords(), blocker = useBlocker(({currentLocation,nextLocation})=>{
    const current=new URLSearchParams(currentLocation.search),next=new URLSearchParams(nextLocation.search);
    current.delete('requestkey');next.delete('requestkey');current.sort();next.sort();
    return dirty&&(currentLocation.pathname!==nextLocation.pathname||current.toString()!==next.toString());
  });
  const message = w('Discard unsaved changes?', 'Отменить несохранённые изменения?');
  const prompted = useRef(false);
  useEffect(() => {
    if (blocker.state !== 'blocked') { prompted.current = false; return; }
    if (prompted.current) return;
    prompted.current = true;
    if (window.confirm(message)) blocker.proceed(); else blocker.reset();
  }, [blocker, message]);
  useEffect(() => {
    const beforeUnload = (event: BeforeUnloadEvent) => { if (dirty) { event.preventDefault(); event.returnValue = ''; } };
    window.addEventListener('beforeunload', beforeUnload);
    return () => window.removeEventListener('beforeunload', beforeUnload);
  }, [dirty]);
  return null;
}

export function formatTime(value: string | null | undefined, locale='en') {
  if (!value || !Number.isFinite(Date.parse(value))) return '—';
  return new Intl.DateTimeFormat(locale, {dateStyle:'medium', timeStyle:'short', timeZone:'UTC'}).format(new Date(value));
}

export const deniesRead = (error: unknown) => error instanceof ApiError && [401,403,404].includes(error.status ?? 0);

export function CollectionLibrary({title,actions,filters,onReset,children}:{title:string;actions?:ReactNode;filters?:ReactNode;onReset?:()=>void;children:ReactNode}) {
  const w=useWords(),popup=useLibraryFilterPopup();
  return <aside className="navigator-library"><header className="panel-head"><h2>{title}</h2><div className="work-library-actions">{actions}
    {filters&&<button ref={popup.trigger} type="button" className="library-icon-action" aria-label={w('Filters','Фильтры')} title={w('Filters','Фильтры')} aria-expanded={popup.open} onClick={popup.toggle}><Filter aria-hidden="true"/></button>}</div></header>
    {filters&&<section ref={popup.popup} popover="auto" role="dialog" aria-label={w('Filters','Фильтры')} className="library-filter-popup strategy-filter-popover" onToggle={event=>popup.setOpen(event.newState==='open')}
      onKeyDown={event=>{if(event.key==='Escape'){event.preventDefault();popup.close();}}}><div className="strategy-filters">{filters}<button type="button" className="library-filter-reset" onClick={onReset}>{w('Reset filters','Сбросить фильтры')}</button></div></section>}
    <div className="library-body">{children}</div></aside>;
}
