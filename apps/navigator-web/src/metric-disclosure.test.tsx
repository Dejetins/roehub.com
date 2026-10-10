import '@testing-library/jest-dom/vitest';
import {cleanup,fireEvent,render,screen} from '@testing-library/react';
import {afterEach,expect,it} from 'vitest';
import {MetricDisclosure} from './metric-disclosure';
afterEach(cleanup);
it('dismisses outside and with Escape, restoring focus without swallowing outside actions',()=>{
 render(<><MetricDisclosure label="P&L" value="100" valueClass="" disclosureLabel="P&L breakdown"><p>Breakdown</p></MetricDisclosure><button>Outside</button></>);
 const trigger=screen.getByRole('button',{name:'P&L breakdown'});
 fireEvent.click(trigger);
 expect(screen.getByRole('region',{name:'P&L breakdown'})).toBeVisible();
 fireEvent.pointerDown(screen.getByRole('button',{name:'Outside'}));
 expect(screen.queryByRole('region')).not.toBeInTheDocument();
 expect(trigger).toHaveAttribute('aria-expanded','false');
 fireEvent.click(trigger);
 fireEvent.keyDown(document,{key:'Escape'});
 expect(screen.queryByRole('region')).not.toBeInTheDocument();
 expect(trigger).toHaveFocus();
});
