import '@testing-library/jest-dom/vitest';
import {afterEach,expect,it} from 'vitest';
import {cleanup,render,screen} from '@testing-library/react';
import {I18nextProvider} from 'react-i18next';
import {InputReadiness} from './backtest-readiness';
import {createI18n} from './i18n';
import type {Preflight} from './builder-api';
afterEach(cleanup);
function mount(data:Preflight['input_readiness']){render(<I18nextProvider i18n={createI18n('en')}><InputReadiness data={data}/></I18nextProvider>);}
it('keeps older-server missing readiness explicitly unknown',()=>{mount(undefined);expect(screen.getByText('Readiness unavailable from this server.')).toBeVisible();expect(screen.queryByText('Ready for worker validation')).not.toBeInTheDocument();});
it.each(['ready','requires_materialization'] as const)('presents %s without claiming payload or backtest completion',status=>{mount({status,requested_signal_rows:2,missing_signal_rows:status==='ready'?0:1,requested_risk_levels:0,risk_coverage:'not_required',estimated_generated_bytes_upper_bound:256,payload_validation:'pending'});expect(screen.getByText(status==='ready'?'Ready for worker validation':'Requires on-demand preparation')).toBeVisible();expect(screen.getByText(/worker verifies payload checksums/)).toBeVisible();});
