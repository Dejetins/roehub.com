async page=>{
 // Real bounded catalog command and immediate cancellation. No candle backfill.
 const command=await page.request.post('http://localhost:20120/api/market-data/work-requests',{headers:{Origin:'http://localhost:20120'},data:{idempotency_key:await page.evaluate(()=>crypto.randomUUID()),kind:'catalog_refresh',market_id:1}});
 if(command.status()!==200)throw new Error('Admission HTTP '+command.status());const created=await command.json();
 const cancel=await page.request.post('http://localhost:20120/api/market-data/work-requests/'+created.job_id+'/cancel',{headers:{Origin:'http://localhost:20120'}});const result=await cancel.json();
 await page.goto('http://localhost:20120/data/ingestion?job='+created.job_id);await page.locator('.work-kv dd').filter({hasText:'Cancelled'}).waitFor();await page.reload();await page.locator('.work-kv dd').filter({hasText:'Cancelled'}).waitFor();
 return {job_id:created.job_id,initial:created.state,cancel_http:cancel.status(),state:result.state,can_retry:result.can_retry,can_cancel:result.can_cancel,reload:true};
}
