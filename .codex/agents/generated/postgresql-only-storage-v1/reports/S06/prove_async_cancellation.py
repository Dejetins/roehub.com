"""Cancellation before/after counted socket startup, including repeated cancellation."""
import asyncio,json,threading
import trading.platform.postgres_budget as budget
DSN='host=s06-pg port=5432 dbname=roehub user=postgres sslmode=verify-full sslrootcert=/input/root.crt'
async def scenario(phase,double):
 before=budget.reservation_counts()
 entered=threading.Event();release=threading.Event()
 name='reserve' if phase=='prepare' else 'connect_raw'
 original=getattr(budget,name)
 def delayed(*a,**kw):
  value=original(*a,**kw)
  entered.set()
  assert release.wait(3)
  return value
 setattr(budget,name,delayed)
 task=asyncio.create_task(budget.async_connect(DSN))
 try:
  while not entered.is_set():await asyncio.sleep(.005)
  task.cancel()
  if double:
   await asyncio.sleep(0)
   task.cancel()
  try:await task
  except asyncio.CancelledError:pass
  else:raise AssertionError('not cancelled')
 finally:
  release.set();setattr(budget,name,original)
 for _ in range(200):
  if budget.reservation_counts()==before:break
  await asyncio.sleep(.01)
 assert budget.reservation_counts()==before,(phase,double,budget.reservation_counts())
async def main():
 for phase in ['prepare','opening']:
  for double in [False,True]:
   for _ in range(3):await scenario(phase,double)
 print(json.dumps({'tls_single_double_cancel':'passed','prepare_opening_scenarios':12,'no_leaked_or_quarantined_reservations':True}))
asyncio.run(main())
