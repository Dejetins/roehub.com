async page => {
 const results=[];
 await page.evaluate(()=>document.documentElement.style.zoom='');
 const check=async(label)=>{
  await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
  const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
  if(overflow)throw new Error('Page overflow: '+label);
  results.push(label);
  await page.screenshot({path:'.codex/delivery/evidence/roehub-overview-2026-09-26/'+label+'.png',fullPage:true});
 };
 for(const locale of ['ru','en']){
  await page.getByRole('link',{name:locale==='ru'?'Русский':'English',exact:true}).click();
  await page.locator('.overview-summary').waitFor();
  for(const width of [820,1024,1440]){
   await page.setViewportSize({width,height:1000});await check('overview-'+locale+'-'+width);
   if(!(await page.locator('.overview-summary>.overview-metric').last().isVisible()))throw new Error('Missing available funds');
  }
  await page.evaluate(()=>document.documentElement.style.zoom='2');await check('overview-'+locale+'-zoom200');
  await page.evaluate(()=>document.documentElement.style.zoom='');
 }
 await page.setViewportSize({width:390,height:844});await check('overview-en-390');
 await page.setViewportSize({width:1440,height:1000});
 await page.emulateMedia({reducedMotion:'reduce'});
 await page.getByRole('button',{name:'Return',exact:true}).click();
 await page.getByRole('button',{name:'Create portfolio',exact:true}).click();
 if(!(await page.getByLabel('Portfolio name',{exact:true}).evaluate(el=>el===document.activeElement)))throw new Error('Dialog autofocus');
 await page.keyboard.press('Escape');
 if(await page.locator('dialog').isVisible())throw new Error('Escape did not close dialog');
 await page.getByRole('button',{name:'Equity',exact:true}).focus();await page.keyboard.press('Enter');
 if(await page.getByRole('button',{name:'Equity',exact:true}).getAttribute('aria-pressed')!=='true')throw new Error('Keyboard activation');
 await check('overview-en-reduced-motion');
 await page.emulateMedia({reducedMotion:'no-preference'});
 await page.getByRole('link',{name:'Русский',exact:true}).click();await page.locator('.overview-summary').waitFor();
 await page.screenshot({path:'.codex/delivery/evidence/roehub-overview-2026-09-26/overview-final.png',fullPage:true});
 return {result:'passed',layouts:results,keyboard:'dialog autofocus, Escape and Enter',reducedMotion:true};
}
