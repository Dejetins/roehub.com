async page=>{
 let writes=0;await page.getByRole('button',{name:'Check saved request',exact:true}).waitFor();
 if(!await page.getByRole('button',{name:'Start download',exact:true}).isDisabled())throw new Error('unknown command unlocked after reload');
 const response=await page.request.get('http://localhost:20120/api/market-data/work-requests/f956150f-52dc-4f14-8bde-0e4304bdae5a');if(response.status()!==200)throw new Error('known job missing');const job=await response.json();
 try{await page.route('**/api/market-data/work-requests',async r=>{if(r.request().method()==='POST'){writes++;await r.abort('failed');}else await r.continue();});await page.route('**/api/market-data/work-requests/lookup?*',r=>r.fulfill({json:job}));await page.getByRole('button',{name:'Check saved request',exact:true}).click();await page.locator('.work-kv dd').filter({hasText:'Completed'}).waitFor();if(writes!==0||page.url().includes('requestkey'))throw new Error('replayed or unreconciled');return {boundary:'lookup response intercepted; reload and explicit reconciliation only',lockedAfterReload:true,replayedWrites:writes,restoredJob:job.job_id};}finally{await page.unrouteAll({behavior:'wait'});}
}
