import {useEffect, useRef, useState, type FormEvent, type ReactNode} from 'react';
import {Link, useLocation} from 'react-router';
import {useInfiniteQuery, useQuery, useQueryClient} from '@tanstack/react-query';
import {useTranslation} from 'react-i18next';
import {Bell, Settings2, Shield, User} from 'lucide-react';
import {z} from 'zod';
import {ApiError} from './api';
import {accountRead, accountSave, auditSchema, notificationDraft, notificationsSchema, preferencesDraft, preferencesSchema, profileDraft, profileSchema, sessionsSchema,
  validateProfile, type Notifications, type NotificationsDraft, type Preferences, type PreferencesDraft, type Profile, type ProfileDraft} from './account-api';
import {applyDisplayPreferences} from './account-preferences';
import {useReadSnapshot} from './read-snapshot';
import {NavigatorTable} from './navigator-table';
import {deniesRead, DirtyGuard, Field, formatTime, ReadFeedback, useWords, WorkError, WorkHeading} from './workspace-ui';

export function SettingsPage({subject}: {subject:string}) {
  const w=useWords(), {pathname}=useLocation();
  const category=pathname.split('/')[2]??'profile';
  const categories=[{id:'profile',label:w('Profile','Профиль'),icon:User},{id:'preferences',label:w('Interface','Интерфейс'),icon:Settings2},
    {id:'notifications',label:w('Notifications','Уведомления'),icon:Bell},{id:'security',label:w('Security','Безопасность'),icon:Shield}];
  return <><aside className="navigator-library"><header className="panel-head"><h2>{w('Settings','Настройки')}</h2></header>
    <nav className="work-categories" aria-label={w('Settings categories','Разделы настроек')}>{categories.map(c=><Link key={c.id} to={`/settings/${c.id}`} aria-current={c.id===category?'page':undefined}><c.icon aria-hidden="true" />{c.label}</Link>)}</nav></aside>
    <WorkHeading title={categories.find(c=>c.id===category)?.label??w('Settings','Настройки')} />
    <div className="work-content" key={category}>{category==='profile'?<ProfileSettings subject={subject}/>:category==='preferences'?<PreferenceSettings subject={subject}/>:category==='notifications'?<NotificationSettings subject={subject}/>:<SecuritySettings subject={subject}/>}</div></>;
}

type EditProps<T,D>={subject:string;path:string;cachePath?:string;className?:string;schema:z.ZodType<T>;draft:(value:T)=>D;validate?:(value:D)=>Record<string,string>;
  saved?:(value:T)=>void;children:(value:D,set:(value:D)=>void,errors:Record<string,string>,source:T)=>ReactNode};
function AccountEditor<T,D extends object>({subject,path,cachePath=path,className='',schema,draft,validate,saved,children}:EditProps<T,D>) {
  const w=useWords(), client=useQueryClient();
  const query=useQuery({queryKey:['account',subject,cachePath],queryFn:({signal})=>accountRead(path,schema,signal),retry:false,refetchOnWindowFocus:false});
  const [accessLost,setAccessLost]=useState(false);
  const denied=deniesRead(query.error)||accessLost;
  const snapshot=useReadSnapshot(subject,path,query.isSuccess?query.data:undefined,denied);
  const [value,setValue]=useState<D>(), [base,setBase]=useState<D>(), [pending,setPending]=useState(false), [didSave,setDidSave]=useState(false);
  const [errors,setErrors]=useState<Record<string,string>>({}), [failure,setFailure]=useState<unknown>(null), [unknown,setUnknown]=useState(false);
  const submitting=useRef(false), form=useRef<HTMLFormElement>(null), dirty=value!==undefined&&JSON.stringify(value)!==JSON.stringify(base);
  useEffect(()=>{
    if(query.isSuccess && query.data!==undefined && !dirty && !pending){const next=draft(query.data);setValue(next);setBase(next);}
  },[query.data,query.isSuccess,dirty,pending,draft]);
  useEffect(()=>{if(denied){setValue(undefined);setBase(undefined);setErrors({});}},[denied]);
  async function submit(event:FormEvent) {
    event.preventDefault();
    if(!value || !dirty || submitting.current || pending || unknown || denied || snapshot.retained) return;
    const found=validate?.(value)??{};
    setErrors(found);setFailure(null);
    if(Object.keys(found).length){requestAnimationFrame(()=>form.current?.querySelector<HTMLElement>(`[name="${Object.keys(found)[0]}"]`)?.focus());return;}
    submitting.current=true;setPending(true);setDidSave(false);
    try {
      const result=await accountSave(path,schema,value), next=draft(result);
      client.setQueryData(['account',subject,cachePath],result);setBase(next);setValue(next);setDidSave(true);saved?.(result);
    } catch(error) {
      setFailure(error);
      if(deniesRead(error)){setAccessLost(true);client.removeQueries({queryKey:['account',subject,cachePath],exact:true});}
      if(error instanceof ApiError){setUnknown(error.outcome==='unknown');setErrors(Object.fromEntries(error.issues.map(issue=>[issue.path.split('.').at(-1)??issue.path,'server'])));}
    } finally {submitting.current=false;setPending(false);}
  }
  async function refresh(){const result=await query.refetch();if(result.isSuccess){setUnknown(false);setFailure(null);setAccessLost(false);}}
  return <section className={`panel ${className}`}>
    <ReadFeedback pending={query.isFetching} initial={!snapshot.data} retained={snapshot.retained} error={query.error??(accessLost?failure:null)}/>
    {(query.error || accessLost || unknown) && <div className="work-actions work-recovery"><button type="button" disabled={pending || query.isFetching} onClick={()=>void refresh()}>{unknown?w('Check saved state','Проверить сохранение'):w('Try again','Повторить')}</button></div>}
    {snapshot.data!==undefined && value!==undefined && <form ref={form} className="work-form" onSubmit={submit} noValidate>
      <DirtyGuard dirty={dirty || pending}/><fieldset disabled={pending || denied || snapshot.retained || unknown} className="work-fields">{children(value,next=>{setValue(next);setDidSave(false);},errors,snapshot.data)}</fieldset>
      <WorkError error={failure}/><div className="work-actions"><button type="submit" className="primary" disabled={!dirty || pending || unknown || snapshot.retained}>{pending?w('Saving…','Сохранение…'):w('Save changes','Сохранить')}</button>
      <button type="button" disabled={!dirty || pending} onClick={()=>{setValue(base);setErrors({});setFailure(null);}}>{w('Cancel','Отмена')}</button>
      <span className="work-save-status" role="status">{pending?w('Saving…','Сохранение…'):unknown?w('Result unknown','Результат неизвестен'):dirty?w('Unsaved changes','Есть несохранённые изменения'):didSave?w('Saved','Сохранено'):''}</span></div>
    </form>}</section>;
}

function ProfileSettings({subject}:{subject:string}) {
  const w=useWords();
  return <AccountEditor<Profile,ProfileDraft> subject={subject} path="profile" schema={profileSchema} draft={profileDraft} validate={validateProfile}>
    {(value,set,errors)=>Object.entries({username:w('Display name','Имя'),email:w('Email','Электронная почта'),timezone:w('Time zone','Часовой пояс'),telegram_discord:w('Contact','Контакт')}).map(([name,label])=>{
      const key=name as keyof ProfileDraft, message=errors[name]?errors[name]==='timezone'?w('Use an IANA time zone, e.g. Europe/Moscow.','Укажите часовой пояс IANA, например Europe/Moscow.'):errors[name]==='email'?w('Enter a valid email address.','Укажите корректный адрес почты.'):w('Check this value.','Проверьте значение.'):undefined;
      return <Field key={name} name={name} label={label} error={message}><input id={name} name={name} type={name==='email'?'email':'text'} autoComplete={name==='username'?'nickname':name==='email'?'email':'off'}
        value={value[key]??''} maxLength={name==='username'||name==='timezone'?80:160} aria-invalid={!!message} aria-describedby={message?`${name}-error`:undefined}
        onChange={e=>set({...value,[name]:e.target.value})}/></Field>;
    })}
  </AccountEditor>;
}

function PreferenceSettings({subject}:{subject:string}) {
  const w=useWords(),{i18n}=useTranslation();
  const [locale,setLocale]=useState<string>();
  useEffect(()=>{if(locale){window.location.assign(`/locale?${new URLSearchParams({locale,next:'/settings/preferences'})}`);}},[locale]);
  return <AccountEditor<Preferences,PreferencesDraft> subject={subject} path="preferences" className="work-preferences-panel" schema={preferencesSchema} draft={preferencesDraft}
    validate={(value):Record<string,string>=>value.autorefresh_preset==='custom'&&(!Number.isInteger(value.refresh_interval_seconds)||value.refresh_interval_seconds<10||value.refresh_interval_seconds>1800)?{refresh_interval_seconds:'range'}:{}}
    saved={value=>{applyDisplayPreferences(value);if(value.locale!==i18n.language)setLocale(value.locale);}}>
    {(value,set,errors,source)=><>
      <Field name="locale" label={w('Language','Язык')}><select id="locale" name="locale" value={value.locale} onChange={e=>set({...value,locale:e.target.value as 'ru'|'en'})}><option value="en">English</option><option value="ru">Русский</option></select></Field>
      <Field name="density" label={w('Table density','Плотность таблиц')}><select id="density" value={value.density} onChange={e=>set({...value,density:e.target.value as Preferences['density']})}><option value="compact">{w('Compact','Компактная')}</option><option value="comfortable">{w('Comfortable','Свободная')}</option></select></Field>
      <Field name="theme" label={w('Theme','Тема')}><select id="theme" value={value.theme} onChange={e=>set({...value,theme:e.target.value as Preferences['theme']})}><option value="graphite">Graphite</option><option value="terminal-orange">Orange</option><option value="matrix-green">Green</option><option value="high-contrast">{w('High contrast','Высокий контраст')}</option></select></Field>
      <Field name="autorefresh_preset" label={w('Automatic refresh','Автообновление')}><select id="autorefresh_preset" value={value.autorefresh_preset} onChange={e=>{const preset=e.target.value;set({...value,autorefresh_preset:preset,refresh_interval_seconds:({off:0,'10s':10,'15s':15,'30s':30,'1m':60,'5m':300} as Record<string,number>)[preset]??value.refresh_interval_seconds});}}>{[...source.autorefresh.allowed_presets,'custom'].map(p=><option key={p} value={p}>{p==='off'?w('Off','Выключено'):p==='custom'?w('Custom','Свой интервал'):p}</option>)}</select></Field>
      {value.autorefresh_preset==='custom'&&<Field name="refresh_interval_seconds" label={w('Interval · seconds','Интервал · секунды')} error={errors.refresh_interval_seconds?w('Enter 10–1800 seconds.','Укажите от 10 до 1800 секунд.'):undefined}><input id="refresh_interval_seconds" name="refresh_interval_seconds" type="number" min={source.autorefresh.min_custom_interval_seconds} max={source.autorefresh.max_custom_interval_seconds} value={value.refresh_interval_seconds} aria-invalid={!!errors.refresh_interval_seconds} aria-describedby={errors.refresh_interval_seconds?'refresh_interval_seconds-error':undefined} onChange={e=>set({...value,refresh_interval_seconds:Number(e.target.value)})}/></Field>}
    </>}
  </AccountEditor>;
}

function NotificationSettings({subject}:{subject:string}) {
  const w=useWords();
  const modes:Record<string,string>={off:w('Off','Выключены'),critical_only:w('Critical only','Только критичные'),signals:w('Signals','Сигналы'),trades:w('Trades','Сделки'),reports:w('Reports','Отчёты'),all:w('All','Все')};
  return <AccountEditor<Notifications,NotificationsDraft> subject={subject} path="notifications/scoped" schema={notificationsSchema} draft={notificationDraft}>
    {(value,set,_errors,source)=><>
      <dl className="work-kv work-wide"><dt>Telegram</dt><dd>{source.telegram_binding.is_confirmed?w('Connected','Подключён'):w('Not connected','Не подключён')}</dd><dt>{w('Delivery route','Доставка')}</dt><dd>{source.route_status}</dd><dt>{w('Sent in 24 hours','Отправлено за сутки')}</dt><dd>{source.delivery_counters.telegram_sent_last_24h}</dd></dl>
      <Field name="mode" label={w('Notifications','Уведомления')}><select id="mode" value={value.mode} onChange={e=>set({...value,mode:e.target.value})}>{source.available_modes.map(mode=><option key={mode} value={mode}>{modes[mode]??mode}</option>)}</select></Field>
      <Field name="timezone" label={w('Report time zone','Часовой пояс отчётов')}><input id="timezone" value={value.timezone} onChange={e=>set({...value,timezone:e.target.value})}/></Field>
      <label className="work-check"><input type="checkbox" checked={value.weekly_enabled} onChange={e=>set({...value,weekly_enabled:e.target.checked})}/>{w('Weekly report','Еженедельный отчёт')}</label>
      <label className="work-check"><input type="checkbox" checked={value.monthly_enabled} onChange={e=>set({...value,monthly_enabled:e.target.checked})}/>{w('Monthly report','Ежемесячный отчёт')}</label>
    </>}
  </AccountEditor>;
}

function SecuritySettings({subject}:{subject:string}) {
  const w=useWords();
  const [tab,setTab]=useState<'sessions'|'audit'>('sessions');
  const history=useAccountHistory(subject,tab);
  return <NavigatorTable value={tab} onChange={id=>setTab(id as 'sessions'|'audit')} expandable={false} tabs={[
    {id:'sessions',label:w('Sessions','Сессии'),tools:tab==='sessions'?history.status:null,content:tab==='sessions'?history.content:null},
    {id:'audit',label:w('Activity','События'),tools:tab==='audit'?history.status:null,content:tab==='audit'?history.content:null},
  ]}/>;
}

function useAccountHistory(subject:string,kind:'sessions'|'audit') {
  const w=useWords(),{i18n}=useTranslation(),sentinel=useRef<HTMLDivElement>(null);
  const query=useInfiniteQuery({queryKey:['account',subject,kind],initialPageParam:'',maxPages:5,retry:false,
    queryFn:async({signal,pageParam}):Promise<{items:(z.infer<typeof sessionsSchema>['items'][number]|z.infer<typeof auditSchema>['items'][number])[];next_cursor:string|null}>=>kind==='sessions'?accountRead(`sessions?limit=20${pageParam?`&cursor=${encodeURIComponent(pageParam)}`:''}`,sessionsSchema,signal):accountRead(`audit-events?limit=20${pageParam?`&cursor=${encodeURIComponent(pageParam)}`:''}`,auditSchema,signal),
    getNextPageParam:page=>page.next_cursor??undefined,
  });
  const history=useReadSnapshot(`${subject}:${kind}`,kind,query.isSuccess?query.data:undefined,deniesRead(query.error)), rows=history.data?.pages.flatMap(p=>p.items)??[];
  useEffect(()=>{const el=sentinel.current;if(!el)return;const observer=new IntersectionObserver(entries=>{if(entries.some(e=>e.isIntersecting)&&query.hasNextPage&&!query.isFetching&&!query.error)void query.fetchNextPage();},{root:el.closest('.work-scroll')});observer.observe(el);return()=>observer.disconnect();},[kind,query.hasNextPage,query.isFetching,query.error,query.fetchNextPage]);
  return {status:<ReadFeedback pending={query.isFetching} retained={history.retained} initial={!history.data&&!query.error}/>,content:<>
    {query.error && <div className="work-history-status"><WorkError error={query.error}/><button type="button" disabled={query.isFetching} onClick={()=>void query.refetch()}>{w('Try again','Повторить')}</button></div>}
    <div className="work-scroll"><table><thead><tr>{(kind==='sessions'?[w('Created · UTC','Создана · UTC'),w('Last active · UTC','Последняя активность · UTC'),w('Expires · UTC','Истекает · UTC'),w('Status','Состояние')]:[w('Time · UTC','Время · UTC'),w('Event','Событие')]).map(label=><th key={label}>{label}</th>)}</tr></thead>
      <tbody>{rows.map(row=>'session_id'in row?<tr key={row.session_id}><td>{formatTime(row.created_at,i18n.language)}</td><td>{formatTime(row.last_seen_at,i18n.language)}</td><td>{formatTime(new Date(Math.min(Date.parse(row.idle_expires_at),Date.parse(row.absolute_expires_at))).toISOString(),i18n.language)}</td><td>{row.revoked_at?w('Revoked','Отозвана'):Math.min(Date.parse(row.idle_expires_at),Date.parse(row.absolute_expires_at))<=Date.now()?w('Expired','Истекла'):w('Active','Активна')}</td></tr>:<tr key={row.event_id}><td>{formatTime(row.created_at,i18n.language)}</td><td>{row.summary}</td></tr>)}</tbody></table>
      {!rows.length&&!query.isPending&&!query.error&&<p className="work-empty">{w('No records','Нет записей')}</p>}<div ref={sentinel}/></div></>};
}
