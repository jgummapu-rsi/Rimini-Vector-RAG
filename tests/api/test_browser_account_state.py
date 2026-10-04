import re
import subprocess
from pathlib import Path


def test_browser_signout_clears_evidence_and_aborts_inflight_requests():
    page = Path("app/api/static/trace/index.html").read_text()
    function = re.search(r"function signOut\(msg\) \{.*?\n\}", page, re.S).group()
    script = (
        """
const assert = require('node:assert/strict');
let accountEpoch=0, token='A', askHistory=[{answer:'private'}], askSelected=2;
let selected='docA', renderedDoc='docA', poll=null;
const TOKEN_KEY='token';
let aborted=false;
const activeRequests=new Set([{abort(){aborted=true;}}]);
const removed=[];
const localStorage={removeItem(key){removed.push(key);}};
const elements=new Map();
const $=id=>{if(!elements.has(id)) elements.set(id,{value:'private',replaceChildren(){this.cleared=true;}});return elements.get(id);};
const showGate=()=>{};
const sourceCleanup=[];
const inlineSourceCleanup=()=>sourceCleanup.push('inline');
const closeSource=()=>sourceCleanup.push('modal');
"""
        + function
        + """
signOut('');
assert.equal(accountEpoch,1);
assert.equal(token,'');
assert.equal(aborted,true);
assert.deepEqual(sourceCleanup,['inline','modal']);
assert.deepEqual(askHistory,[]);
assert.equal(selected,null);
assert.equal(renderedDoc,null);
assert(removed.includes('rag_ask_history'));
assert.equal($('askdetail').cleared,true);
assert.equal($('askq').value,'');
"""
    )
    subprocess.run(["node", "-e", script], check=True, capture_output=True, timeout=10)
    assert "res.status === 401 || res.status === 403" not in page
    assert "localStorage.setItem(ASK_HIST_KEY" not in page
