import {useState} from 'react';
import {useInfiniteQuery, useQueryClient} from '@tanstack/react-query';
import {useTranslation} from 'react-i18next';
import {Monitor, RefreshCw} from 'lucide-react';
import {z} from 'zod';
import {requestJson} from './api';
import {accountRead} from './account-api';
import {useReadSnapshot} from './read-snapshot';
import {deniesRead, ReadFeedback, useWords, WorkError} from './workspace-ui';

const deviceSchema=z.object({device_id:z.string(),label:z.string(),last_seen_at:z.string(),expires_at:z.string(),active:z.boolean(),is_current:z.boolean()});
export const devicesSchema=z.object({items:z.array(deviceSchema),next_cursor:z.string().nullable(),legacy_active_count:z.number()});
const revokedSchema=z.object({revoked:z.boolean(),current:z.boolean()});

export function SessionDevices({subject}:{subject:string}) {
  const w=useWords(),{i18n}=useTranslation(),client=useQueryClient();
  const [pending,setPending]=useState<string|null>(null),[failure,setFailure]=useState<unknown>(null);
  const [uncertain,setUncertain]=useState(false);
  const query=useInfiniteQuery({queryKey:['account',subject,'devices'],initialPageParam:'',maxPages:5,retry:false,refetchInterval:60000,
    queryFn:({signal,pageParam})=>accountRead(`devices?limit=20${pageParam?`&cursor=${encodeURIComponent(pageParam)}`:''}`,devicesSchema,signal),
    getNextPageParam:page=>page.next_cursor??undefined,
  });
  const snapshot=useReadSnapshot(subject,'devices',query.isSuccess?query.data:undefined,deniesRead(query.error)||deniesRead(failure));
  const rows=snapshot.data?.pages.flatMap(page=>page.items)??[];
  const legacy=snapshot.data?.pages[0]?.legacy_active_count??0;
  const locked=Boolean(pending)||snapshot.retained||Boolean(query.error)||uncertain;
  const time=(value:string)=>new Intl.DateTimeFormat(i18n.language,{dateStyle:'short',timeStyle:'short',timeZone:'UTC'}).format(new Date(value));
  async function refresh() {
    const result=await query.refetch();
    if(result.isSuccess){setUncertain(false);setFailure(null);}
  }
  async function revoke(id:string) {
    if(locked)return;
    setPending(id);setFailure(null);
    try {
      const {data}=await requestJson(`/api/ui/account/devices/${encodeURIComponent(id)}/revoke`,revokedSchema,{method:'POST',body:{},csrf:true});
      if(data.current){client.clear();window.location.assign('/login');return;}
      await refresh();
    } catch(error) {setFailure(error);setUncertain(true);if(deniesRead(error)){await client.cancelQueries({queryKey:['account',subject,'devices']});client.removeQueries({queryKey:['account',subject,'devices']});}}
    finally {setPending(null);}
  }
  return <section className="security-devices" aria-label={w('Devices','Устройства')}>
    <div className="security-device-intro"><div>
      <p>{w('Browsers signed in to your account. A valid sign-in does not mean the device is online.','Браузеры, в которых выполнен вход в ваш аккаунт. Действующий вход не означает, что устройство сейчас онлайн.')}</p>
      <small>{w('One row per browser profile. Different browsers and private windows are separate devices. Activity updates within a minute.','Одна строка на профиль браузера. Другой браузер или приватное окно — отдельное устройство. Активность обновляется в течение минуты.')}</small>
    </div><button type="button" className="icon-button" aria-label={w('Refresh devices','Обновить устройства')} disabled={Boolean(pending)||query.isFetching} onClick={()=>void refresh()}><RefreshCw size={16}/></button></div>
    <ReadFeedback pending={query.isFetching} retained={snapshot.retained} initial={!snapshot.data&&!query.error}/>
    <WorkError error={query.error??failure}/>
    {uncertain&&<p role="status">{w('Refresh the list to check the result before trying again.','Обновите список, чтобы проверить результат перед повторной попыткой.')}</p>}
    <div className="work-scroll"><table className="security-device-table"><thead><tr>
      <th>{w('Device','Устройство')}</th><th>{w('Last activity · UTC','Последняя активность · UTC')}</th><th>{w('Sign-in','Вход')}</th><th><span className="sr-only">{w('Actions','Действия')}</span></th>
    </tr></thead><tbody>{rows.map(row=><tr key={row.device_id}>
      <td><div className="security-device-name"><Monitor size={16} aria-hidden="true"/><strong>{row.label==='Browser'?w('Unknown browser','Неизвестный браузер'):row.label}</strong>{row.is_current&&<span className="security-device-current">{w('This device','Это устройство')}</span>}</div></td>
      <td>{time(row.last_seen_at)}</td><td><span className={row.active?'security-device-valid':''}>{row.active?w('Valid','Действителен'):w('Ended','Завершён')}</span></td>
      <td>{row.active&&<button type="button" disabled={locked} onClick={()=>void revoke(row.device_id)} aria-label={row.is_current?w('Sign out on this device','Выйти на этом устройстве'):w(`End sign-in: ${row.label}`,`Завершить вход: ${row.label}`)}>{pending===row.device_id?w('Ending…','Завершение…'):row.is_current?w('Sign out','Выйти'):w('End sign-in','Завершить вход')}</button>}</td>
    </tr>)}</tbody></table></div>
    {!rows.length&&query.isSuccess&&<p className="work-empty">{w('No devices','Нет устройств')}</p>}
    {query.hasNextPage&&<button type="button" disabled={query.isFetching||Boolean(pending)} onClick={()=>void query.fetchNextPage()}>{w('Show more','Показать ещё')}</button>}
    {legacy>0&&<div className="security-legacy-sessions"><div><strong>{w('Earlier sign-ins','Ранние входы')} · {legacy}</strong><p>{w('These sign-ins predate device tracking. Their browsers and last activity are unknown. They can all be ended without signing out here.','Для этих входов браузер и последняя активность неизвестны: сведения ещё не собирались. Их можно завершить вместе, сохранив вход на этом устройстве.')}</p></div><button type="button" disabled={locked} onClick={()=>void revoke('legacy')}>{pending==='legacy'?w('Ending…','Завершение…'):w('End earlier sign-ins','Завершить старые входы')}</button></div>}
  </section>;
}
