import {z} from 'zod';
import {requestJson} from './api';

export const profileSchema = z.object({user_id:z.string(), username:z.string().nullable(), email:z.string().nullable(),
  timezone:z.string(), locale:z.enum(['ru','en']), telegram_discord:z.string().nullable(), subscription_status:z.string(), updated_at:z.string()});
export const preferencesSchema = z.object({theme:z.enum(['graphite','terminal-orange','matrix-green','high-contrast']), locale:z.enum(['ru','en']),
  density:z.enum(['compact','comfortable']), autorefresh:z.object({preset_key:z.string(), refresh_interval_seconds:z.number(),
    allowed_presets:z.array(z.string()), min_custom_interval_seconds:z.number(), max_custom_interval_seconds:z.number()}), updated_at:z.string()});
export const notificationsSchema = z.object({telegram_binding:z.object({is_confirmed:z.boolean(), chat_id_ref_masked:z.string().nullable(), confirmed_at:z.string().nullable()}),
  mode:z.string(), route_status:z.string(), recipient_address_ref_masked:z.string().nullable(),
  report_schedule:z.object({weekly_enabled:z.boolean(), monthly_enabled:z.boolean(), timezone:z.string()}),
  delivery_counters:z.object({telegram_sent_total:z.number(), telegram_sent_last_24h:z.number(), last_telegram_sent_at:z.string().nullable()}),
  available_modes:z.array(z.string()), updated_at:z.string()});
export const sessionsSchema = z.object({items:z.array(z.object({session_id:z.string(), created_at:z.string(), last_seen_at:z.string(),
  idle_expires_at:z.string(), absolute_expires_at:z.string(), revoked_at:z.string().nullable(), is_current:z.boolean()})), next_cursor:z.string().nullable()});
export const auditSchema = z.object({items:z.array(z.object({event_id:z.string(), created_at:z.string(), event_type:z.string(), summary:z.string()})), next_cursor:z.string().nullable()});
export type Profile = z.infer<typeof profileSchema>;
export type Preferences = z.infer<typeof preferencesSchema>;
export type Notifications = z.infer<typeof notificationsSchema>;
export type ProfileDraft = Pick<Profile,'username'|'email'|'timezone'|'telegram_discord'>;
export type PreferencesDraft = Pick<Preferences,'theme'|'locale'|'density'> & {autorefresh_preset:string; refresh_interval_seconds:number};
export type NotificationsDraft = {mode:string; weekly_enabled:boolean; monthly_enabled:boolean; timezone:string};

export async function accountRead<T>(path:string, schema:z.ZodType<T>, signal?:AbortSignal) {
  return (await requestJson(`/api/ui/account/${path}`,schema,{signal})).data;
}
export async function accountSave<T>(path:string, schema:z.ZodType<T>, body:unknown) {
  return (await requestJson(`/api/ui/account/${path}`,schema,{method:'PUT',body})).data;
}

export function validateProfile(value:ProfileDraft):Record<string,string> {
  const errors:Record<string,string>={};
  if ((value.username?.length ?? 0)>80) errors.username='length';
  if ((value.email?.length ?? 0)>160 || (value.email && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value.email))) errors.email='email';
  if ((value.telegram_discord?.length ?? 0)>160) errors.telegram_discord='length';
  try { new Intl.DateTimeFormat('en',{timeZone:value.timezone}); if (!value.timezone.trim()) errors.timezone='timezone'; }
  catch { errors.timezone='timezone'; }
  return errors;
}

export const profileDraft = (p:Profile):ProfileDraft => ({username:p.username??'',email:p.email??'',timezone:p.timezone,telegram_discord:p.telegram_discord??''});
export const preferencesDraft = (p:Preferences):PreferencesDraft => ({theme:p.theme,locale:p.locale,density:p.density,autorefresh_preset:p.autorefresh.preset_key,refresh_interval_seconds:p.autorefresh.refresh_interval_seconds});
export const notificationDraft = (p:Notifications):NotificationsDraft => ({mode:p.mode,...p.report_schedule});
