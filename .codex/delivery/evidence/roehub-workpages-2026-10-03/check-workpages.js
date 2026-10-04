async page => {
 const routes=[['profile','/settings/profile','Display name'],['preferences','/settings/preferences','Theme'],['notifications','/settings/notifications',null],['security','/settings/security',null],['connections','/connections?source=1',null],['data','/data?market=1&symbol=BTCUSDT&from=2024-01-01T00%3A00&to=2024-01-01T00%3A05',null],['ingestion','/data/ingestion?job=f956150f-52dc-4f14-8bde-0e4304bdae5a',null],['monitoring','/monitoring/api',null]];
 const results=[],errors=[],httpFailures=[];const onError=e=>errors.push(e.message.slice(0,250));const onResponse=r=>{if(r.status()>=400)httpFailures.push({path:r.url().split('?')[0].replace(/^https?:\/\/[^/]+/,''),status:r.status()});};page.on('pageerror',onError);page.on('response',onResponse);
 for(const width of [1476,390]){
  await page.setViewportSize({width,height:959});
  for(const [name,path,label] of routes){
   await page.goto('http://localhost:20120'+path);await page.locator('#workspace-heading').waitFor();
   if(label)await page.getByLabel(label,{exact:true}).waitFor();
   else await page.waitForFunction(()=>!document.querySelector('.work-content .loading-data'));
   if(name==='monitoring')await page.locator('.work-kv dd').filter({hasText:'Application readiness confirmed'}).waitFor();
   if(name==='ingestion')await page.locator('.work-kv dd').filter({hasText:'Completed'}).waitFor();
   if(name==='data'){await page.locator('.work-object-row').first().waitFor();await page.getByText('No gaps in this period',{exact:true}).waitFor();}
   if(name==='security')await page.locator('.navigator-table tbody tr').first().waitFor();
   if(name==='connections')await page.getByText('Last successful read · UTC',{exact:true}).waitFor();
   await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
   await page.screenshot({path:'.codex/delivery/evidence/roehub-workpages-2026-10-03/'+name+'-'+width+'.png',fullPage:true});
   const measure=await page.evaluate(()=>{
    const selectors=['.sidebar','.navigator-library','.navigator-library>.panel-head','.navigator-heading','.navigator-heading h1','.library-icon-action','.library-icon-action svg','.navigator-table','.navigator-table>.overview-toolbar','.navigator-table th','.navigator-table td','input:not([type=checkbox])','button[role=tab]'];
    return {width:innerWidth,scrollWidth:document.documentElement.scrollWidth,elements:selectors.flatMap(selector=>{const el=document.querySelector(selector);if(!el)return [];const c=getComputedStyle(el),r=el.getBoundingClientRect();return [{selector,rect:{x:r.x,y:r.y,width:r.width,height:r.height},css:Object.fromEntries(['fontFamily','fontSize','fontWeight','lineHeight','padding','gap','border','borderRadius','backgroundColor','color'].map(k=>[k,c[k]]))}];})};
   });
   if(measure.scrollWidth>width+1)throw new Error(name+' overflow '+width+': '+measure.scrollWidth);
   results.push({name,...measure});
  }
 }
 await page.setViewportSize({width:1476,height:959});
 page.off('pageerror',onError);page.off('response',onResponse);return {results,errors,httpFailures};
}
