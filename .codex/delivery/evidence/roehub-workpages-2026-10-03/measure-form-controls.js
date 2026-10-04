async page=>{
 const props=['fontFamily','fontSize','fontWeight','lineHeight','padding','border','borderRadius','backgroundColor','color','outline','outlineOffset'];
 const result=[];await page.setViewportSize({width:1476,height:959});
 for(const [path,selectors] of [['backtests/new',['.builder input[name=strategy_name]','.builder select[name="coordinates.exchange"]','.builder .field:has(>input[type=number])>label','.builder button[type=submit]']],['settings/profile',['.work-form input[name=username]','.work-field:has(input[name=username])>label','.work-form button[type=submit]']],['settings/preferences',['.work-form select#locale','.work-field:has(select#locale)>label','.work-form button[type=submit]']]]){
  await page.goto('http://localhost:20120/'+path);await page.locator(selectors[0]).waitFor();await page.mouse.move(0,0);const fields=[];
  for(const selector of selectors){const el=page.locator(selector).first();if(await el.count())fields.push({selector,...await el.evaluate((e,props)=>{const s=getComputedStyle(e),r=e.getBoundingClientRect();return {width:r.width,height:r.height,css:Object.fromEntries(props.map(p=>[p,s[p]]))};},props)});}
  result.push({path,fields});
 }
 return result;
}
