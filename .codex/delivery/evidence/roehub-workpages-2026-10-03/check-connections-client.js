async page=>{
 // Every private connection endpoint is intercepted. Public catalog/auth reads remain real.
 const base='/api/ui/account/exchange-connections',id='00000000-0000-4000-8000-000000000099',now=new Date().toISOString();
 let row={connection_id:id,exchange_name:'binance',market_type:'spot',environment:'testnet',label:'Client-only QA connection',requested_permissions:'read',effective_permissions:'none',effective_capability:'none',connection_readiness:'needs_action',connection_readiness_reason:'skipped_external_validation',status:'active',validation_status:'skipped_external_validation',validation_reason:null,last_validated_at:null,created_at:now,updated_at:now,used_by_strategies_count:0,active_strategy_bindings_count:0};
 let exists=false,denyDisable=true;const commands=[],proof=[];
 await page.route('**'+base+'**',async r=>{const u=r.request().url(),method=r.request().method();
  if(method==='GET'){if(u.includes('/bindings'))return r.fulfill({json:{items:[],next_cursor:null}});if(u.includes('/'+id))return r.fulfill({json:row});const q=u.includes('q=absent-client-row')?'absent-client-row':'';return r.fulfill({json:{items:exists&&row.label.toLowerCase().includes(q.toLowerCase())?[row]:[],next_cursor:null}});}
  const action=u.endsWith('/validate')?'validate':u.endsWith('/rotate')?'rotate':u.endsWith('/disable')?'disable':u.endsWith('/archive')?'archive':'create';commands.push(action);
  if(action==='disable'&&denyDisable)return r.fulfill({status:409,json:{error:{code:'connection_in_use'}}});
  if(action==='create')exists=true;
  if(action==='validate')row={...row,validation_status:'valid_readonly',effective_permissions:'read',last_validated_at:now};
  if(action==='disable')row={...row,status:'disabled',connection_readiness:'disconnected'};
  if(action==='archive')row={...row,status:'archived',connection_readiness:'archived'};
  await r.fulfill({json:row});
 });
 try{
  await page.goto('http://localhost:20120/connections');await page.getByText('No connections',{exact:true}).waitFor();await page.getByRole('button',{name:'New connection',exact:true}).click();await page.getByLabel('Name',{exact:true}).fill(row.label);
  await page.getByLabel('API key',{exact:true}).fill('qa-placeholder-not-a-key');await page.getByLabel('API secret',{exact:true}).fill('qa-placeholder-not-a-secret');await page.getByRole('button',{name:'Save connection',exact:true}).click();await page.getByRole('heading',{name:row.label,exact:true}).waitFor();await page.getByText('No bindings in this organization',{exact:true}).waitFor();
  if(commands.join()!=='create')throw new Error('create count');await page.getByRole('button',{name:'Check connection',exact:true}).click();await page.getByText('Read access verified',{exact:true}).waitFor();proof.push('create/check distinct active, validated and action-required readiness');
  await page.getByRole('button',{name:'Replace credentials',exact:true}).click();if(await page.getByLabel('API key',{exact:true}).inputValue()||await page.getByLabel('API secret',{exact:true}).inputValue())throw new Error('stored secret displayed');await page.getByLabel('API key',{exact:true}).fill('qa-replacement-placeholder');await page.getByLabel('API secret',{exact:true}).fill('qa-replacement-placeholder');await page.getByRole('button',{name:'Save connection',exact:true}).click();await page.getByText('Read access verified',{exact:true}).waitFor();proof.push('replacement form has empty secret fields; explicit client-only replacement');
  await page.getByRole('button',{name:'Disconnect',exact:true}).click();await page.getByRole('dialog').getByRole('button',{name:'Disconnect',exact:true}).click();await page.getByRole('alert').waitFor();if(row.status!=='active')throw new Error('dependency denial ignored');denyDisable=false;
  await page.getByRole('button',{name:'Disconnect',exact:true}).click();await page.getByRole('dialog').getByRole('button',{name:'Disconnect',exact:true}).click();await page.locator('.work-kv dd').filter({hasText:'Disconnected'}).first().waitFor();await page.getByRole('button',{name:'Archive',exact:true}).click();await page.getByRole('dialog').getByRole('button',{name:'Archive',exact:true}).click();await page.locator('.work-kv dd').filter({hasText:'Archived'}).first().waitFor();proof.push('server dependency conflict preserved active state, explicit disconnect/archive with confirmations');
  await page.getByRole('button',{name:'Filters',exact:true}).click();await page.getByLabel('Search',{exact:true}).fill('absent-client-row');await page.getByText('No connections',{exact:true}).waitFor();await page.getByRole('button',{name:'Reset filters',exact:true}).click();await page.keyboard.press('Escape');await page.reload();await page.locator('.work-kv dd').filter({hasText:'Archived'}).first().waitFor();proof.push('search/empty/reset and direct selected URL reload');
  const leaked=await page.evaluate(()=>[location.href,JSON.stringify(localStorage),JSON.stringify(sessionStorage),document.body.innerText].some(v=>v.includes('qa-placeholder')||v.includes('qa-replacement')));if(leaked)throw new Error('placeholder leak');
  return {boundary:'intercepted client contract only; zero private provider writes; no real credentials',commands,proof,placeholderLeak:false};
 }finally{await page.unrouteAll({behavior:'wait'});await page.goto('http://localhost:20120/connections?source=1');}
}
