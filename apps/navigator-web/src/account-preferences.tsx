import {useEffect} from 'react';
import {useQuery} from '@tanstack/react-query';
import {useTranslation} from 'react-i18next';
import {accountRead, preferencesSchema, type Preferences} from './account-api';

export function applyDisplayPreferences(preferences: Preferences) {
  document.documentElement.dataset.navigatorTheme = preferences.theme;
  document.documentElement.dataset.navigatorDensity = preferences.density;
}
export function AccountPreferences({subject}: {subject:string}) {
  const {i18n}=useTranslation();
  const query=useQuery({queryKey:['account',subject,'preferences'], queryFn:({signal})=>accountRead('preferences',preferencesSchema,signal),retry:false});
  useEffect(()=>{if(query.data) applyDisplayPreferences(query.data);},[query.data]);
  // Locale navigation is server-owned. A saved preference explicitly updates that same locale.
  useEffect(()=>{document.documentElement.lang=i18n.language;},[i18n.language]);
  return null;
}
