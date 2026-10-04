import {afterEach,expect,it} from 'vitest';
import {cleanup,renderHook} from '@testing-library/react';
import {useReadSnapshot} from './read-snapshot';
afterEach(cleanup);
it('retains one complete bundle, replaces it atomically, and clears on access loss or subject change',()=>{
 const first={metric:1,series:[1]},second={metric:2,series:[2]};
 const {result,rerender}=renderHook(({scope,key,data,blocked}:{scope:string;key:string;data:typeof first|undefined;blocked:boolean})=>useReadSnapshot(scope,key,data,blocked),{initialProps:{scope:'a',key:'one',data:first as typeof first|undefined,blocked:false}});
 rerender({scope:'a',key:'two',data:undefined,blocked:false});expect(result.current).toMatchObject({data:first,retained:true,displayedKey:'one'});
 rerender({scope:'a',key:'two',data:second,blocked:false});expect(result.current).toMatchObject({data:second,retained:false,displayedKey:'two'});
 rerender({scope:'a',key:'two',data:undefined,blocked:true});expect(result.current.data).toBeUndefined();
 rerender({scope:'a',key:'two',data:undefined,blocked:false});expect(result.current.data).toBeUndefined();
 rerender({scope:'a',key:'one',data:first,blocked:false});
 rerender({scope:'b',key:'one',data:undefined,blocked:false});expect(result.current.data).toBeUndefined();
});
