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
// Chunk outlines use the full source extent, while quote highlights stay narrow.
context.document = {createElementNS: (_, tag) => ({tag, attrs: {}, children: [],
  setAttribute(name, value) { this.attrs[name] = value; },
  append(...children) { this.children.push(...children); }})};
vm.runInContext(html.slice(html.indexOf('function passageOverlay('), html.indexOf('function renderSourceText(')), context);
const overlay = context.passageOverlay([line(.2, .3, .1)], 1, false,
  [{page: 1, x: .1, y: .1, width: .7, height: .5},
   {page: 2, x: .1, y: .1, width: .8, height: .8}]);
assert.equal(overlay.children.length, 2);
assert.equal(overlay.children[0].attrs.class, 'chunk-outline');
assert.equal(overlay.children[0].attrs.x, 96);
assert.equal(overlay.children[0].attrs.height, 508);
assert.equal(overlay.children[1].attrs.x, 199.5);
assert.equal(context.passageOverlay([], 1, true, []).children.length, 0);
const edge = context.passageOverlay([], 1, true,
  [{page: 1, x: 0, y: 0, width: 1, height: 1}]).children[0];
assert.equal(edge.attrs.x, 0);
assert.equal(edge.attrs.width, 1000);
assert(!html.includes('id="inlineSourceBody"'));
assert(html.includes('dialog.show()'));
assert(html.includes('.answer-citation, .citation .open-source'));
assert(html.includes('signal: controller.signal'));
console.log('Side preview: syntax, grouping, page isolation and placement passed');
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
const citationContext = vm.createContext({esc: text => String(text).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')});
vm.runInContext(html.slice(html.indexOf('function answerWithCitations('), html.indexOf('function renderAskDetail(')), citationContext);
const rendered = citationContext.answerWithCitations('Use [encrypt]. Sources [2] [17] [2]. <img src=x> [99]',
  [{source_id: '17'}, {source_id: '2'}]);
assert.equal((rendered.match(/class="answer-citation"/g) || []).length, 3);
assert(rendered.includes('data-citation="1"'));
assert(rendered.includes('data-citation="0"'));
assert(rendered.includes('[encrypt]'));
assert(rendered.includes('[99]'));
assert(!rendered.includes('<img'));
assert(!citationContext.answerWithCitations('No evidence [1]', []).includes('<button'));
assert(rendered.includes('data-occurrence="2"'));
const bound = {source_id: '3', supporting_quote: 'Wrong generic quote', provenance: {regions: [{page: 5}]},
  occurrences: [{occurrence: 1, supporting_quotes: ['Lost devices'], provenance: {regions: [{page: 5, y: .2}]}},
    {occurrence: 2, supporting_quotes: ['Report immediately', 'security@example.com'], provenance: {regions: [{page: 5, y: .5}]}}]};
assert.equal(citationContext.citationForOccurrence(bound, 1).supporting_quote, 'Lost devices');
assert(citationContext.citationForOccurrence(bound, 2).supporting_quote.includes('security@example.com'));
assert.equal(citationContext.citationForOccurrence(bound, 2).provenance.regions[0].y, .5);
assert.equal(citationContext.citationForOccurrence(bound, 3).provenance.regions.length, 0);
assert.equal(citationContext.citationForOccurrence({...bound, occurrences: []}, 1).supporting_quote, '');
console.log('Inline citation links: ID mapping, repeats, literal brackets, and escaping passed');
