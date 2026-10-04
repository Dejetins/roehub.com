import '@testing-library/jest-dom/vitest';
import {afterEach,expect,it,vi} from 'vitest';
import {act,cleanup,render,screen} from '@testing-library/react';
import {createMemoryRouter,RouterProvider} from 'react-router';
import {QueryClientProvider} from '@tanstack/react-query';
import {I18nextProvider} from 'react-i18next';
import {MonitoringPage} from './monitoring-page';
import {createQueryClient} from './query-client';
import {createI18n} from './i18n';

const service={service_id:'private-worker',group:'compute',capability:'compute.jobs',state:'healthy',observed_at:new Date().toISOString(),detail_code:'probe.domain_ready',signal_source:'operational-health',runbook_path:null,required:true};
afterEach(()=>{cleanup();vi.restoreAllMocks();vi.unstubAllGlobals();});
it.each([401,403,404])('handles detail denial %s against a cached inventory',async status=>{
 vi.stubGlobal('fetch',vi.fn(url=>Promise.resolve(String(url).endsWith('/ui/monitoring')?Response.json({generated_at:service.observed_at,groups:['compute'],services:[service],events:[],history_state:'available',history_scope:'process',commands:[]}):Response.json({error:{code:'denied'}},{status}))));
 const client=createQueryClient(),router=createMemoryRouter([{path:'*',element:<MonitoringPage subject="owner"/>}],{initialEntries:['/monitoring']});
 render(<I18nextProvider i18n={createI18n('en')}><QueryClientProvider client={client}><RouterProvider router={router}/></QueryClientProvider></I18nextProvider>);
 await screen.findAllByText('private worker');
 await act(()=>router.navigate('/monitoring/private-worker'));
 await screen.findAllByText(status===401?'Your session has ended.':status===403?'Access denied.':'The resource is unavailable or no longer exists.');
 if(status===404)expect(screen.getByText('private worker')).toBeVisible();
 else expect(screen.queryByText('private worker')).not.toBeInTheDocument();
});
