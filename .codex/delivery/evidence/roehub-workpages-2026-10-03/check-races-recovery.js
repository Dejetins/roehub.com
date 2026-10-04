async page=>{
 const proof=[],dialogs=[];const onDialog=async d=>{dialogs.push(d.type());await d.dismiss();};page.on('dialog',onDialog);
 const url='http://localhost:20120/data?market=1&symbol=BTCUSDT&q=ETHUSDT&from=2024-01-01T00%3A00&to=2024-01-01T00%3A05';let release;
 try{
  await page.goto(url);await page.getByRole('heading',{name:'BTCUSDT',exact:true}).waitFor();await page.locator('.work-object-row').filter({hasText:'ETHUSDT'}).waitFor();
  let intercepted=false;const gate=new Promise(r=>{release=r;});
  await page.route('**/api/market-data/workspace/catalog?*',async r=>{if(r.request().url().includes('q=ETHUSDT&limit=1')){intercepted=true;await gate;try{await r.continue();}catch{}}else await r.continue();});
  await page.locator('.work-object-row').filter({hasText:'ETHUSDT'}).click();await page.getByText('Showing previous data',{exact:true}).first().waitFor();
  if(!intercepted||!await page.getByRole('heading',{name:'BTCUSDT',exact:true}).isVisible()||!await page.getByRole('button',{name:'Select instrument',exact:true}).isDisabled())throw new Error('retained identity/command lock');
  await page.goBack();release();await page.getByRole('heading',{name:'BTCUSDT',exact:true}).waitFor();await page.unrouteAll({behavior:'wait'});await page.reload();await page.getByRole('heading',{name:'BTCUSDT',exact:true}).waitFor();proof.push('delayed ETH detail retains BTC identity, command locked, back/reload correct; delayed response cannot replace selection');
  await page.getByRole('link',{name:'Download candles',exact:true}).click();await page.getByLabel('Symbols · comma separated',{exact:true}).waitFor();if(await page.getByLabel('Symbols · comma separated',{exact:true}).inputValue()!=='BTCUSDT')throw new Error('data-to-download selection');
  await page.getByLabel('Symbols · comma separated',{exact:true}).fill('BTCUSDT, BTCUSDT');await page.getByRole('button',{name:'Start download',exact:true}).click();await page.getByText('Choose 1–8 distinct symbols and no more than 10,080 closed candles.',{exact:true}).waitFor();
  await page.getByLabel('Symbols · comma separated',{exact:true}).fill('BTCUSDT');let posts=0,key='';
  await page.route('**/api/market-data/work-requests',async r=>{if(r.request().method()==='POST'){posts++;key=r.request().postDataJSON().idempotency_key;await r.abort('failed');}else await r.continue();});
  await page.getByRole('button',{name:'Start download',exact:true}).click();await page.getByRole('button',{name:'Check saved request',exact:true}).waitFor();if(posts!==1||!await page.getByRole('button',{name:'Start download',exact:true}).isDisabled())throw new Error('duplicate guard');
  const recoveryURL=page.url();if(!recoveryURL.includes('requestkey='+key))throw new Error('missing URL recovery key');if(dialogs.length)throw new Error('unexpected dirty guard on recovery key');proof.push('client validation, unknown submission locked, URL recovery key saved; command aborted before server');
  return {boundary:'network delay/abort intercepted: frontend proof; no candle command reached server',proof,commandCount:posts,recoveryURL};
 }finally{release?.();page.off('dialog',onDialog);await page.unrouteAll({behavior:'wait'});}
}
