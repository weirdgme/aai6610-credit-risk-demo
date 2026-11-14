import {predict, validateModel, verifyIntegrity} from './model.js';

const fields = [
  ['LIMIT_BAL','Credit limit (NT$)','number','profile-fields'],
  ['AGE','Age in years','number','profile-fields'],
  ['SEX','Historical sex code',[[1,'1 · Male'],[2,'2 · Female']],'profile-fields'],
  ['EDUCATION','Education code',[[0,'0 · Unspecified'],[1,'1 · Graduate school'],[2,'2 · University'],[3,'3 · High school'],[4,'4 · Other'],[5,'5 · Other recorded code'],[6,'6 · Other recorded code']],'profile-fields'],
  ['MARRIAGE','Marital status code',[[0,'0 · Unspecified'],[1,'1 · Married'],[2,'2 · Single'],[3,'3 · Other']],'profile-fields'],
];
['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6'].forEach((name,i) => fields.push([name,`Month ${i+1} status (${name})`,Array.from({length:12},(_,j)=>[j-2,`${j-2}${j-2===-1?' · Paid duly':j-2>=1?' · Months delayed':' · Historical code'}`]),'status-fields']));
for(let i=1;i<=6;i++) { fields.push([`BILL_AMT${i}`,`Month ${i} bill (NT$)`,'number','amount-fields']); fields.push([`PAY_AMT${i}`,`Month ${i} payment (NT$)`,'number','amount-fields']); }
for(const [name,label,type,section] of fields) {
  const group=document.createElement('div'); const caption=document.createElement('label'); caption.htmlFor=name; caption.textContent=label;
  const control=document.createElement(type==='number'?'input':'select'); control.id=name; control.name=name;
  if(type==='number') { control.type='number'; control.step='1'; control.inputMode='decimal'; if(name==='AGE'){control.min='18';control.max='100';} }
  else for(const [value,text] of type) { const option=document.createElement('option'); option.value=String(value);option.textContent=text;control.append(option); }
  group.append(caption,control); document.getElementById(section).append(group);
}
let model,examples,manifest;
function loadExample(i) { const row=examples.cases[i].input; for(const [name] of fields) document.getElementById(name).value=row[name]===null?'':String(row[name]); document.getElementById('error').hidden=true;document.getElementById('result').hidden=true; }
document.getElementById('example').addEventListener('change',e=>loadExample(Number(e.target.value)));
document.getElementById('reset').addEventListener('click',()=>loadExample(Number(document.getElementById('example').value)));
document.getElementById('profile-form').addEventListener('submit',event=>{
  event.preventDefault(); const result=document.getElementById('result'); const error=document.getElementById('error');result.hidden=true;error.hidden=true;
  try {
    const input=Object.fromEntries(fields.map(([name])=>{const value=document.getElementById(name).value.trim();return[name,value===''?null:Number(value)];}));
    const output=predict(input,model);result.replaceChildren();
    const label=document.createElement('p');label.className='score-label';label.textContent='Historical default score';
    const score=document.createElement('p');score.className='score';score.textContent=`${(output.score*100).toFixed(2)}%`;
    const classification=document.createElement('p');classification.className='classification';classification.textContent=`Model class: ${output.class?'Default':'Nondefault'}`;
    const note=document.createElement('p');note.className='result-note';note.textContent='Fixed threshold: 0.5. This historical model output is not a credit approval or a present-day risk assessment.';
    result.append(label,score,classification,note);
    if(output.warnings.length){const list=document.createElement('ul');list.className='warnings';for(const warning of output.warnings){const li=document.createElement('li');li.textContent=warning;list.append(li);}result.append(list);}
    result.hidden=false;
  } catch(e) { error.textContent=e.message;error.hidden=false; }
});
try {
  const [response,m,fixtures]=await Promise.all([fetch('./model.json',{cache:'no-cache'}),fetch('./manifest.json',{cache:'no-cache'}),fetch('./fixtures.json')]);
  if(!response.ok||!m.ok||!fixtures.ok)throw new Error('Model files could not be loaded. Please try again.');
  const bytes=await response.arrayBuffer();manifest=await m.json();await verifyIntegrity(bytes,manifest);model=JSON.parse(new TextDecoder().decode(bytes));validateModel(model);examples=await fixtures.json();
  if(model.model_version!==manifest.model_version)throw new Error('Model version does not match the release.');
  document.getElementById('model-info').textContent=`Version ${model.model_version}. Fixed fitted architecture: 33 inputs, 64 and 32 hidden units, 1 output. Historical test accuracy 81.90%; default recall 35.42%. The test set was examined in earlier assignments and is not a new independent evaluation.`;
  loadExample(0);document.getElementById('calculate').disabled=false;document.getElementById('loading').textContent='Model loaded. Choose an example or enter invented values.';
} catch(e) { document.getElementById('loading').textContent='Scoring is unavailable.';const error=document.getElementById('error');error.textContent=e.message;error.hidden=false; }
