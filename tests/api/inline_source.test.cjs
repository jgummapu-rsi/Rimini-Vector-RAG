const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('app/api/static/trace/index.html', 'utf8');
new vm.Script(html.match(/<script>([\s\S]*?)<\/script>/)[1]);
const grouping = html.slice(html.indexOf('function passageGroups('), html.indexOf('function passageOverlay('));
const context = vm.createContext({});
vm.runInContext(grouping, context);
const group = lines => JSON.parse(JSON.stringify(context.passageGroups(lines, 1)));
const line = (x, y, width = .7) => ({page: 1, x, y, width, height: .02});
assert.equal(group([line(.1, .1), line(.1, .124)]).length, 2);
assert.equal(group([line(.1, .1), line(.1, .3)]).length, 2);
assert.equal(group([line(.1, .1, .3), line(.6, .124, .3)]).length, 2);
assert.equal(group([{...line(.1, .1), page: 2}]).length, 0);
assert.equal(group([line(NaN, .1), line(.9, .1)]).length, 0);
assert(Math.abs(group([line(.1, .1), line(.1, .124, .3)])[0].height - .02) < 1e-9);
assert.equal(group([{...line(.1, .1, .1), precision: 'word'},
  {...line(.205, .1, .1), precision: 'word'}]).length, 1);
assert.equal(group([{...line(.1, .1, .1), precision: 'word', layout_id: 1},
  {...line(.205, .1, .1), precision: 'word', layout_id: 2}]).length, 1);
assert.equal(group([{...line(.1, .102, .1), precision: 'word'},
  {...line(.205, .1, .1), precision: 'word'},
  {...line(.31, .102, .1), precision: 'word'}]).length, 1);
assert.equal(group([{...line(.1, .1), precision: 'page_region'}])[0].approximate, true);
assert(html.indexOf('id="inlineSourceBody"') > html.indexOf('<div class="answercard">${esc(entry.answer)}</div>'));
assert(html.includes('signal: controller.signal'));
console.log('Inline preview: syntax, grouping, page isolation and placement passed');
const quoteCode = html.slice(html.indexOf('function uniqueQuoteRange('), html.indexOf('function scrollToHighlight('));
vm.runInContext(quoteCode, context);
const quoteRange = (text, quote) => {
  const range = context.uniqueQuoteRange(text, quote);
  return range ? Array.from(range) : null;
};
assert.deepEqual(quoteRange('Before\nSol is an accent.\nAfter', 'Sol is an accent.'), [7, 24]);
assert.deepEqual(quoteRange('Before\nSol is\nan accent.\nAfter', 'Sol is an accent.'), [7, 24]);
assert.equal(quoteRange('Same quote. Same quote.', 'Same quote.'), null);
assert.equal(quoteRange('Actual source text', 'Invented quote'), null);
assert.equal(quoteRange('Actual source text', ''), null);
console.log('Citation quote matching: original offsets, whitespace, ambiguity, and missing quotes passed');
