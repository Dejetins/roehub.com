async page=>{
 // Network interception proves client error handling only; no provider claim.
 const proof=[];await page.setViewportSize({width:1476,height:959});
 try{
  await page.goto('http://localhost:20120/settings/profile');const name=page.getByLabel('Display name',{exact:true});await name.waitFor();const saved=await name.inputValue();let profileStatus=503;
  await page.route('**/api/ui/account/profile',r=>r.fulfill({status:profileStatus,json:{error:{code:'test_unavailable'}}}));
  await page.getByRole('button',{name:'Refresh',exact:true}).click();await page.getByText('Showing previous data',{exact:true}).waitFor();if(await name.inputValue()!==saved)throw new Error('Profile snapshot lost');
  profileStatus=403;await page.getByRole('button',{name:'Refresh',exact:true}).click();await name.waitFor({state:'hidden'});proof.push('profile503 retain/403 hide');await page.unrouteAll({behavior:'wait'});
  await page.goto('http://localhost:20120/monitoring/api');await page.locator('.work-kv').waitFor();let status=503;
  await page.route('**/api/ui/monitoring/api',r=>r.fulfill({status,json:{error:{code:'test_denial'}}}));await page.getByRole('button',{name:'Refresh',exact:true}).first().click();await page.getByText('Showing previous data',{exact:true}).waitFor();
  if(await page.locator('.work-kv').count()!==1)throw new Error('Monitoring snapshot lost');status=403;await page.getByRole('button',{name:'Refresh',exact:true}).first().click();await page.locator('.work-kv').waitFor({state:'hidden'});if(await page.locator('.work-object-row').count())throw new Error('Protected inventory retained');proof.push('monitoring503 retain/detail403 closes inventory');await page.unrouteAll({behavior:'wait'});
  await page.goto('http://localhost:20120/monitoring/api');await page.locator('.work-kv').waitFor();
  await page.route('**/api/ui/monitoring/api',async r=>{const real=await r.fetch();const body=await real.json();body.service.observed_at='2024-01-01T00:00:00Z';await r.fulfill({json:body});});await page.getByRole('button',{name:'Refresh',exact:true}).last().click();await page.locator('.work-kv dd').filter({hasText:'Stale observation'}).waitFor();proof.push('monitoring stale never healthy');await page.unrouteAll({behavior:'wait'});
  const job='f956150f-52dc-4f14-8bde-0e4304bdae5a';await page.goto('http://localhost:20120/data/ingestion?job='+job);await page.locator('.work-kv dd').filter({hasText:'Completed'}).waitFor();
  await page.route('**/api/market-data/work-requests/'+job,r=>r.fulfill({status:404,json:{error:{code:'not_found'}}}));await page.locator('.work-content').getByRole('button',{name:'Refresh',exact:true}).click();await page.locator('.work-kv').waitFor({state:'hidden'});proof.push('ingestion removed detail hidden');await page.unrouteAll({behavior:'wait'});
 }finally{await page.unrouteAll({behavior:'wait'});}
 await page.goto('http://localhost:20120/monitoring/api');return {boundary:'intercepted failure responses; client only',proof};
}
