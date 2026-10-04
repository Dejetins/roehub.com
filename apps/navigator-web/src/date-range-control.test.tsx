import {afterEach,beforeAll,expect,it,vi} from 'vitest';
import {cleanup,fireEvent,render,screen} from '@testing-library/react';
import {DateRangeControl,dateWindow} from './date-range-control';
vi.mock('react-i18next',()=>({useTranslation:()=>({i18n:{language:'en'}})}));
beforeAll(()=>{
 HTMLElement.prototype.showPopover=function(){this.setAttribute('data-open','true');this.style.display='block';};
 HTMLElement.prototype.hidePopover=function(){this.removeAttribute('data-open');this.style.display='none';};
});
afterEach(cleanup);
const bounds={start:Date.parse('2026-09-01T12:00Z'),end:Date.parse('2026-09-26T12:00Z')};
it('clips inclusive UTC dates to coverage and rejects invalid/reversed/outside ranges',()=>{
 expect(dateWindow('2026-09-01','2026-09-26',bounds)).toEqual(bounds);
 expect(dateWindow('2026-09-10','2026-09-10',bounds)).toEqual({start:Date.parse('2026-09-10T00:00Z'),end:Date.parse('2026-09-10T23:59:59.999Z')});
 for(const [from,to] of [['','2026-09-12'],['2026-09-12','2026-09-11'],['2026-08-31','2026-09-12'],['2026-09-01','2026-09-27']])expect(dateWindow(from!,to!,bounds)).toBeNull();
});
it('does not commit drafts or cancellation; Apply commits once and restores focus',()=>{
 const apply=vi.fn();render(<DateRangeControl bounds={bounds} value={bounds} active={false} onApply={apply}/>);
 const trigger=screen.getByRole('button',{name:'Date range'});
 fireEvent.click(trigger);fireEvent.input(screen.getByLabelText('From'),{target:{value:'2026-09-10'}});
 expect(apply).not.toHaveBeenCalled();fireEvent.click(screen.getByRole('button',{name:'Cancel'}));
 expect(apply).not.toHaveBeenCalled();expect(document.activeElement).toBe(trigger);
 fireEvent.click(trigger);expect((screen.getByLabelText('From') as HTMLInputElement).value).toBe('2026-09-01');
 fireEvent.input(screen.getByLabelText('From'),{target:{value:'2026-09-10'}});
 fireEvent.click(screen.getByRole('button',{name:'Apply'}));
 expect(apply).toHaveBeenCalledExactlyOnceWith({start:Date.parse('2026-09-10'),end:bounds.end});expect(document.activeElement).toBe(trigger);
});
it('Escape cancels and a missing history disables the trigger',()=>{
 const apply=vi.fn();const {rerender}=render(<DateRangeControl bounds={bounds} value={null} active={false} onApply={apply}/>);
 fireEvent.click(screen.getByRole('button',{name:'Date range'}));fireEvent.keyDown(screen.getByLabelText('From'),{key:'Escape'});
 expect(apply).not.toHaveBeenCalled();expect(document.activeElement).toBe(screen.getByRole('button',{name:'Date range'}));
 rerender(<DateRangeControl bounds={null} value={null} active={false} onApply={apply}/>);
 expect((screen.getByRole('button',{name:'Date range'}) as HTMLButtonElement).disabled).toBe(true);
});
