async page => {
  const root = ".codex/delivery/evidence/roehub-workpages-2026-10-03";
  const result = {};
  for (const route of ["backtests", "strategies", "dashboard"]) {
    await page.goto("http://localhost:20120/" + route);
    await page.locator(".navigator-table tbody tr").first().waitFor();
    await page.locator(".result-metrics dd, .operations-metrics strong, .overview-summary strong").first().waitFor();
    await page.locator("canvas").first().waitFor();
    await page.evaluate(() => new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
    await page.screenshot({path: root + "/reference-" + route + ".png"});
    result[route] = await page.evaluate(() => {
      const selectors = [".sidebar", ".navigator-library", ".navigator-library>.panel-head",
        ".navigator-heading, .operations-header", ".navigator-table", ".navigator-table>.overview-toolbar",
        ".navigator-table th", ".navigator-table td", ".library-icon-action",
        ".view-switch>button", ".overview-expand", ".date-range-trigger", ".navigator-chart, .overview-performance"];
      return {viewport:{width:innerWidth,height:innerHeight,dpr:devicePixelRatio},
        elements:selectors.map(selector => {
          const el = document.querySelector(selector); if (!el) return {selector,missing:true};
          const rect = el.getBoundingClientRect(), css = getComputedStyle(el);
          const props = ["fontFamily","fontSize","fontWeight","lineHeight","padding","gap","border","borderRadius","backgroundColor","color","height","width"];
          return {selector,rect:rect.toJSON(),css:Object.fromEntries(props.map(p=>[p,css[p]]))};
        })};
    });
    if (route === "backtests") {
      await page.getByRole("button",{name:"Filters",exact:true}).click();
      await page.screenshot({path:root+"/reference-backtests-filters.png"});
      await page.keyboard.press("Escape");
      await page.getByRole("button",{name:"Download",exact:true}).click();
      await page.screenshot({path:root+"/reference-backtests-download.png"});
      await page.keyboard.press("Escape");
      await page.getByRole("button",{name:"Expand tables",exact:true}).click();
      await page.screenshot({path:root+"/reference-backtests-expanded.png"});
      await page.keyboard.press("Escape");
    }
    if (route === "strategies") {
      await page.getByRole("button",{name:"Filters",exact:true}).click();
      await page.screenshot({path:root+"/reference-strategies-filters.png"});
      await page.keyboard.press("Escape");
    }
  }
  return result;
}
