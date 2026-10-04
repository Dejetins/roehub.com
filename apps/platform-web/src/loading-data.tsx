import { LoaderCircle } from 'lucide-react';
import { useTranslation } from 'react-i18next';

export function LoadingData() {
  const { t } = useTranslation();
  return <span className="loading-data" role="status" aria-atomic="true"><LoaderCircle aria-hidden="true" /><span>{t('loadingData')}</span></span>;
}

/** Reserve geometry even when an update finishes. Previous values stay explicitly labelled. */
export function ReadStatus({pending, retained=false,className=''}: {pending:boolean; retained?:boolean;className?:string}) {
  const {i18n}=useTranslation();
  return <div className={`read-status ${className}`} role="status" aria-live="polite" aria-atomic="true">{pending ? <span>{i18n.language.startsWith('ru')?'Обновление…':'Updating…'}</span> : null}{retained && <span>{i18n.language.startsWith('ru')?'Показаны предыдущие данные':'Showing previous data'}</span>}</div>;
}
