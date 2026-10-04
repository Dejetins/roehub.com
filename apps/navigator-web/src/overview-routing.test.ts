import {expect,it} from 'vitest';
import {clientRoute} from './client-routes';
it('keeps Overview server-owned unless explicitly enabled in bootstrap',()=>{
 expect(clientRoute('/dashboard',{locale:'en',subject:'demo'})).toBe(false);
 expect(clientRoute('/dashboard',{locale:'en',subject:'demo',client_routes:['/backtests','/strategies']})).toBe(false);
 expect(clientRoute('/dashboard',{locale:'en',subject:'demo',client_routes:['/dashboard']})).toBe(true);
 expect(clientRoute('https://other.example/dashboard',{locale:'en',subject:'demo',client_routes:['/dashboard']})).toBe(false);
});
