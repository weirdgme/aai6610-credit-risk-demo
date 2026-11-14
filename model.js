/* Pure inference for the exact fitted Module 8 network. No training or network calls. */
export function validateModel(model) {
  if (model.format_version !== 1 || model.threshold !== 0.5 || model.layers?.length !== 3) throw new Error('Unsupported model format.');
  if (model.feature_order?.length !== 23 || model.numeric_features?.length !== 20 || model.categories?.length !== 3) throw new Error('Invalid feature schema.');
  let width = 33;
  for (const [i, layer] of model.layers.entries()) {
    const outputs = [64, 32, 1][i];
    if (layer.weights.length !== width || layer.bias.length !== outputs ||
        layer.weights.some(row => row.length !== outputs || row.some(v => !Number.isFinite(v))) || layer.bias.some(v => !Number.isFinite(v))) throw new Error('Invalid network dimensions or parameters.');
    width = outputs;
  }
  if (model.numeric_medians.length !== 20 || model.numeric_means.length !== 20 || model.numeric_scales.length !== 20 ||
      [...model.numeric_medians, ...model.numeric_means, ...model.numeric_scales].some(v => !Number.isFinite(v)) || model.numeric_scales.some(v => v <= 0)) throw new Error('Invalid preprocessing parameters.');
  if (model.categories.map(v => v.length).reduce((a,b) => a+b, 0) !== 13) throw new Error('Invalid one-hot dimensions.');
}

export function validateInput(input, model) {
  if (!input || typeof input !== 'object' || Array.isArray(input)) throw new Error('Provide one profile.');
  const expected = new Set(model.feature_order);
  for (const name of Object.keys(input)) if (!expected.has(name)) throw new Error(`Unexpected field ${name}.`);
  const missing = model.feature_order.filter(name => !Object.hasOwn(input, name));
  if (missing.length) throw new Error(`Missing field ${missing[0]}.`);
  const warnings = [];
  for (const name of model.numeric_features) {
    const value = input[name];
    if (value === null) { warnings.push(`${name}: training median used for the blank value.`); continue; }
    if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`${name} must be a finite number or blank.`);
    if (name === 'AGE' && (!Number.isInteger(value) || value < 18 || value > 100)) throw new Error('Age must be a whole number from 18 to 100.');
    if (name.startsWith('PAY_') && !name.startsWith('PAY_AMT') && (!Number.isInteger(value) || value < -2 || value > 9)) throw new Error(`${name} must be a whole status code from -2 to 9.`);
    if ((name === 'LIMIT_BAL' || name.startsWith('PAY_AMT')) && (value < 0 || value > 1e9)) throw new Error(`${name} must be between 0 and 1,000,000,000.`);
    if (name.startsWith('BILL_AMT') && Math.abs(value) > 1e9) throw new Error(`${name} is outside the demonstration input limits.`);
  }
  model.categorical_features.forEach((name, i) => {
    if (!model.categories[i].includes(input[name])) throw new Error(`${name} is not a supported historical category code.`);
  });
  return warnings;
}

export function transform(input, model) {
  const numeric = model.numeric_features.map((name, i) => ((input[name] === null ? model.numeric_medians[i] : input[name]) - model.numeric_means[i]) / model.numeric_scales[i]);
  const oneHot = model.categorical_features.flatMap((name, i) => model.categories[i].map(code => Number(input[name] === code)));
  return [...numeric, ...oneHot];
}

export function predict(input, model) {
  const warnings = validateInput(input, model);
  let values = transform(input, model);
  model.layers.forEach((layer, index) => {
    const sums = [...layer.bias];
    for (let i = 0; i < values.length; i++) for (let j = 0; j < sums.length; j++) sums[j] += values[i] * layer.weights[i][j];
    values = index < model.layers.length - 1 ? sums.map(v => Math.max(0, v)) : sums.map(v => v >= 0 ? 1 / (1 + Math.exp(-v)) : Math.exp(v) / (1 + Math.exp(v)));
  });
  const score = values[0];
  if (!Number.isFinite(score) || score < 0 || score > 1) throw new Error('The model could not produce a valid score.');
  return {score, class: Number(score >= model.threshold), threshold: model.threshold, model_version: model.model_version, warnings};
}

export async function verifyIntegrity(bytes, manifest) {
  const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), v => v.toString(16).padStart(2, '0')).join('');
  if (hash !== manifest.model_sha256) throw new Error('Model integrity check failed. Please reload the application.');
  return hash;
}
