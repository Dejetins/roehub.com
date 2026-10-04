async page=>{
 const output=[];const properties=['fontFamily','fontSize','fontWeight','lineHeight','padding','gap','border','borderRadius','backgroundColor','color','outline','outlineOffset'];
 const measure=async(sel)=>page.locator(sel).first().evaluate((e,props)=>{const r=e.getBoundingClientRect(),s=getComputedStyle(e);return {rect:{x:r.x,y:r.y,width:r.width,height:r.height},css:Object.fromEntries(props.map(p=>[p,s[p]]))};},properties);
 for(const width of [1476,390]){
 await page.setViewportSize({width,height:959});
 for(const path of ['backtests','strategies','dashboard','settings/security','data?market=1&symbol=BTCUSDT&from=2024-01-01T00%3A00&to=2024-01-01T00%3A05','monitoring/api']){
  await page.goto('http://localhost:20120/'+path);await page.locator('.navigator-table th').first().waitFor();if(path.startsWith('data?'))await page.locator('.work-object-row').first().waitFor();await page.mouse.move(0,0);
  const entry={path,width,elements:{},filter:{}};
  for(const selector of ['.sidebar','.navigator-library','.navigator-library>.panel-head','.navigator-heading, .operations-header','.navigator-table>.overview-toolbar','.navigator-table th','.navigator-table td','.navigator-table button[role=tab]'])if(await page.locator(selector).count())entry.elements[selector]=await measure(selector);
  if(await page.getByRole('button',{name:'Filters',exact:true}).count()){
   const b=page.getByRole('button',{name:'Filters',exact:true}),sel='button[aria-label="Filters"]';entry.filter.normal=await measure(sel);await b.hover();await b.evaluate(e=>Promise.all(e.getAnimations().map(a=>a.finished.catch(()=>{}))));entry.filter.hover=await measure(sel);await page.mouse.move(0,0);await b.focus();await page.keyboard.press('Tab');await page.keyboard.press('Shift+Tab');entry.filter.focus=await measure(sel);await b.click();await b.evaluate(async e=>{await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));await Promise.all(e.getAnimations().map(a=>a.finished.catch(()=>{})));});entry.filter.open=await measure(sel);entry.popup=await measure('.library-filter-popup');await page.keyboard.press('Escape');
  }
  output.push(entry);
 }
 }
 await page.setViewportSize({width:1476,height:959});return output;
}
