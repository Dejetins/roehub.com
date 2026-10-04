async page=>{
 const proof=[];
 await page.setViewportSize({width:1476,height:959});
 await page.goto('http://localhost:20120/monitoring/api');await page.locator('.work-kv').waitFor();
 const filters=page.getByRole('button',{name:'Filters',exact:true});await filters.focus();await page.keyboard.press('Enter');await page.getByLabel('Search',{exact:true}).waitFor();await page.getByLabel('Search',{exact:true}).fill('not-a-service');await page.getByText('No matches',{exact:true}).first().waitFor();
 await page.getByRole('button',{name:'Reset filters',exact:true}).click();await page.keyboard.press('Escape');if(!await filters.evaluate(e=>e===document.activeElement))throw new Error('Filter focus not restored');proof.push('monitoring search/reset/keyboard Escape focus');
 const expand=page.getByRole('button',{name:'Expand tables',exact:true});await expand.click();await page.getByRole('dialog',{name:'Tables',exact:true}).waitFor();await page.keyboard.press('Escape');if(!await expand.evaluate(e=>e===document.activeElement))throw new Error('Table focus not restored');proof.push('shared table expansion/Escape focus');
 await page.locator('.work-object-row').filter({hasText:'ClickHouse'}).click();await page.locator('.work-kv dd').filter({hasText:'Port reachable; readiness is not confirmed'}).waitFor();await page.goBack();await page.locator('.work-kv dd').filter({hasText:'Application readiness confirmed'}).waitFor();proof.push('monitoring selection/back URL');
 const runbook=await page.getByRole('link',{name:'Open runbook'}).getAttribute('href');const response=await page.request.get('http://localhost:20120'+runbook);if(response.status()!==200)throw new Error('runbook HTTP '+response.status());proof.push('runbook HTTP200');
 for(const width of [1476,390]){
  await page.setViewportSize({width,height:959});await page.goto('http://localhost:20120/data?market=1&symbol=BTCUSDT&from=2024-01-01T00%3A00&to=2024-01-01T00%3A05');await page.locator('.work-object-row').first().waitFor();await page.getByText('No gaps in this period',{exact:true}).waitFor();
  await page.getByRole('button',{name:'Filters',exact:true}).click();await page.getByLabel('Search symbol',{exact:true}).fill('BTCUSDT');await page.waitForFunction(()=>document.querySelectorAll('.work-object-row').length===1);await page.keyboard.press('Escape');proof.push('catalog bounded search '+width);
  await page.screenshot({path:'.codex/delivery/evidence/roehub-workpages-2026-10-03/data-'+width+'.png',fullPage:true});
 }
 await page.setViewportSize({width:1476,height:959});return proof;
}
