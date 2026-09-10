import '@testing-library/jest-dom/vitest';
import { afterEach, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { I18nextProvider } from 'react-i18next';
import { createI18n } from './i18n';
import { MotionSettings, motionDuration, transitionUI, useDisclosureMotion } from './motion';
afterEach(()=>{cleanup();localStorage.clear();document.documentElement.style.removeProperty('--motion-duration');vi.restoreAllMocks();vi.unstubAllGlobals();});
it('uses one persisted preference for CSS and structural transitions, with reduced-motion priority',()=>{
 render(<I18nextProvider i18n={createI18n('en')}><MotionSettings/></I18nextProvider>);
 fireEvent.change(screen.getByLabelText('Animation'),{target:{value:'slow'}});
 expect(motionDuration()).toBe(520);
 expect(document.documentElement.style.getPropertyValue('--motion-duration')).toBe('520ms');
 vi.stubGlobal('matchMedia',()=>({matches:true}));
 expect(motionDuration()).toBe(0);
 const update=vi.fn();transitionUI(update);expect(update).toHaveBeenCalledTimes(1);
});
it('skips a superseded animation while executing each requested update once',async()=>{
 const skip=vi.fn();let finish!:()=>void;
 const finished=new Promise<void>(resolve=>{finish=resolve;});
 const start=vi.fn((update:()=>void)=>{update();return {ready:Promise.resolve(),finished,skipTransition:skip};});
 vi.stubGlobal('document',Object.assign(document,{startViewTransition:start}));
 const first=vi.fn(),second=vi.fn();
 transitionUI(first,'layout');transitionUI(second,'layout');
 expect(skip).toHaveBeenCalledTimes(1);expect(first).toHaveBeenCalledTimes(1);expect(second).toHaveBeenCalledTimes(1);
 finish();await finished;await Promise.resolve();
 delete (document as Partial<Document>).startViewTransition;
});
it('animates native disclosure toggles without changing their contents',()=>{
 function Example(){useDisclosureMotion();return <div data-platform-client><details><summary>Details</summary><input aria-label="Draft" defaultValue="retained"/></details></div>;}
 render(<Example/>);const input=screen.getByLabelText('Draft');
 fireEvent.click(screen.getByText('Details'));expect(document.querySelector('details')).toHaveAttribute('open');
 fireEvent.click(screen.getByText('Details'));expect(document.querySelector('details')).not.toHaveAttribute('open');
 expect(screen.getByLabelText('Draft')).toBe(input);
});
it('applies rapid content changes immediately and animates inside an expanded surface',()=>{
 const animation={cancel:vi.fn()};const animate=vi.fn(()=>animation);
 const parent=document.createElement('div');parent.setAttribute('data-motion-content','');
 const modal=document.createElement('div');modal.setAttribute('aria-modal','true');
 const content=document.createElement('div');content.setAttribute('data-motion-content','');
 Object.assign(content,{animate});modal.append(content);parent.append(modal);document.body.append(parent);
 const changes:string[]=[];
 transitionUI(()=>changes.push('equity'));transitionUI(()=>changes.push('price'));
 expect(changes).toEqual(['equity','price']);expect(animate).toHaveBeenCalledTimes(2);expect(animation.cancel).toHaveBeenCalledTimes(1);
 parent.remove();
});
