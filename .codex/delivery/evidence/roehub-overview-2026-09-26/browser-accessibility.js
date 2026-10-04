async page => {
 const results=[];
 const audit=async(label)=>{
  await page.evaluate(()=>Promise.all(document.getAnimations().filter(a=>a.effect?.getTiming().iterations!==Infinity).map(a=>a.finished.catch(()=>{}))));
  if(!await page.evaluate(()=>!!window.axe))await page.addScriptTag({path:'node_modules/.pnpm/axe-core@4.13.0/node_modules/axe-core/axe.min.js'});
  const violations=await page.evaluate(async()=> (await axe.run()).violations.map(v=>({id:v.id,impact:v.impact,targets:v.nodes.map(n=>n.target)})));
  if(violations.length)throw new Error(label+': '+JSON.stringify(violations));
  results.push(label);
 };
 await audit('RU ready');
 await page.getByRole('button',{name:'Создать портфель',exact:true}).click();
 await audit('RU editor');
 await page.getByRole('button',{name:'Сохранить портфель',exact:true}).click();await audit('RU validation');
 await page.keyboard.press('Escape');
 await page.getByRole('button',{name:'Развернуть график',exact:true}).click();
 await page.evaluate(()=>Promise.all(document.getAnimations().map(a=>a.finished.catch(()=>{}))));
 const bounds=await page.locator('.overview-expanded').boundingBox();
 if(!bounds||bounds.x>15||bounds.y>15||bounds.width<1400||bounds.height<960)throw new Error('Fullscreen does not cover viewport: '+JSON.stringify(bounds));
 await audit('RU expanded chart');
 const last=page.locator('.overview-expanded summary').last();await last.focus();await page.keyboard.press('Tab');
 if(!(await page.locator('.overview-expanded').evaluate(el=>el.contains(document.activeElement))))throw new Error('Fullscreen focus escaped');
 await page.keyboard.press('Escape');
 await page.locator('.overview-expanded').waitFor({state:'hidden'});
 if(await page.locator('[inert]').count())throw new Error('Inert was not restored');
 await page.getByRole('link',{name:'English',exact:true}).click();await page.locator('.overview-summary').waitFor();await audit('EN ready');
 await page.setViewportSize({width:820,height:1000});await audit('EN 820');
 await page.setViewportSize({width:1440,height:1000});
 await page.getByRole('link',{name:'Русский',exact:true}).click();await page.locator('.overview-summary').waitFor();await audit('RU final');
 await page.screenshot({path:'.codex/delivery/evidence/roehub-overview-2026-09-26/overview-final.png',fullPage:true});
 return {result:'passed',axe:results,fullscreen:'viewport bounds, focus containment, inert restoration'};
}
