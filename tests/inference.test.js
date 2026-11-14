import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {predict, transform, validateInput, validateModel, verifyIntegrity} from '../model.js';

const bytes=await readFile(new URL('../model.json',import.meta.url));
const model=JSON.parse(bytes);
const manifest=JSON.parse(await readFile(new URL('../manifest.json',import.meta.url)));
const fixtures=JSON.parse(await readFile(new URL('../fixtures.json',import.meta.url)));
const clone=()=>structuredClone(fixtures.cases[0].input);

test('release matches the original fitted model version and checksum',async()=>{
  validateModel(model);
  assert.equal(model.model_version,'module8-candidate10-epoch12-v1');
  assert.equal(model.model_version,manifest.model_version);
  assert.equal(createHash('sha256').update(bytes).digest('hex'),manifest.model_sha256);
  assert.equal(await verifyIntegrity(bytes,manifest),manifest.model_sha256);
  assert.equal(model.layers.reduce((sum,l)=>sum+l.weights.flat().length+l.bias.length,0),4289);
});
for(const [i,fixture] of fixtures.cases.entries()){
  test(`original Python parity on synthetic profile ${i+1}`,()=>{
    const output=predict(fixture.input,model);
    assert.ok(Math.abs(output.score-fixture.score)<1e-10,`score ${output.score} != ${fixture.score}`);
    assert.equal(output.class,fixture.class);
    assert.ok(output.score>=0&&output.score<=1);
    assert.equal(output.threshold,.5);
    assert.equal(output.model_version,manifest.model_version);
  });
}
test('field insertion order does not change the prediction',()=>{
  const input=clone();const shuffled=Object.fromEntries(Object.entries(input).reverse());
  assert.deepEqual(predict(input,model),predict(shuffled,model));
});
test('missing numeric values use the original fitted medians and issue warnings',()=>{
  const input=clone();input.AGE=null;input.BILL_AMT3=null;
  const output=predict(input,model);
  assert.equal(output.warnings.length,2);
  const explicit={...input,AGE:model.numeric_medians[model.numeric_features.indexOf('AGE')],BILL_AMT3:model.numeric_medians[model.numeric_features.indexOf('BILL_AMT3')]};
  assert.ok(Math.abs(output.score-predict(explicit,model).score)<1e-12);
  assert.equal(transform(input,model).length,33);
});
test('duplicate synthetic requests return identical results',()=>{
  assert.deepEqual(predict(clone(),model),predict(clone(),model));
});
for(const [label,mutate] of [
  ['missing field',x=>delete x.AGE], ['extra identifier',x=>x.ID=42],
  ['unknown sex code',x=>x.SEX=3], ['unknown education code',x=>x.EDUCATION=99],
  ['unknown marriage code',x=>x.MARRIAGE=99], ['missing categorical value',x=>x.SEX=null],
  ['nonfinite value',x=>x.LIMIT_BAL=Infinity],['NaN value',x=>x.AGE=NaN],
  ['string numeric value',x=>x.AGE='35'],['implausible age',x=>x.AGE=9],
  ['fractional age',x=>x.AGE=35.2],['invalid repayment code',x=>x.PAY_0=99],
  ['negative payment amount',x=>x.PAY_AMT1=-5],['unbounded bill value',x=>x.BILL_AMT2=1e100]
])test(`rejects ${label}`,()=>{const input=clone();mutate(input);assert.throws(()=>validateInput(input,model));});
test('negative bill balances and recorded unknown category codes are retained',()=>{
  const input={...clone(),BILL_AMT1:-500,EDUCATION:0,MARRIAGE:0};
  assert.doesNotThrow(()=>predict(input,model));
});
test('tampered model is rejected before inference',async()=>{
  const corrupted=Buffer.from(bytes);corrupted[corrupted.length-5]^=1;
  await assert.rejects(verifyIntegrity(corrupted,manifest),/integrity/);
});
test('invalid weight dimensions are rejected',()=>{
  const invalid=structuredClone(model);invalid.layers[0].weights.pop();
  assert.throws(()=>validateModel(invalid),/dimensions/);
});
test('zero scaler is rejected',()=>{
  const invalid=structuredClone(model);invalid.numeric_scales[0]=0;
  assert.throws(()=>validateModel(invalid),/preprocessing/);
});
test('browser form never transmits profile values or persists them',async()=>{
  const script=await readFile(new URL('../app.js',import.meta.url),'utf8');
  assert.ok(script.includes('event.preventDefault()'));
  assert.ok(!script.includes('localStorage')&&!script.includes('sessionStorage'));
  assert.equal((script.match(/fetch\(/g)||[]).length,3);
  assert.ok(!script.includes('method:')&&!script.includes('sendBeacon'));
});
test('release README uses free static hosting and a real HTML entry point',async()=>{
  const readme=await readFile(new URL('../README.md',import.meta.url),'utf8');
  assert.match(readme,/sdk: static/);assert.match(readme,/app_file: index.html/);
  const html=await readFile(new URL('../index.html',import.meta.url),'utf8');
  assert.match(html,/type="module" src="app.js"/);
});
