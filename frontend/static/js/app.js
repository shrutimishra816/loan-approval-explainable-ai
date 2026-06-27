/* app.js — LoanSight frontend logic */

const API_BASE = '';   // same origin; update if running frontend separately

// ── Demo data ──────────────────────────────────────────────────────────────
const DEMO_DATA = {
  age: 35,
  income: 600000,
  coapplicant_income: 200000,
  loan_amount: 500000,
  loan_term: '36',
  credit_score: 720,
  employment_yrs: 5,
  existing_loans: 1,
  num_dependents: 2,
  education_grad: 1,   // Graduate
  self_employed: 0,    // Salaried
  area: 'semi',        // Semi-Urban
};

// ── Feature label map ──────────────────────────────────────────────────────
const FEATURE_LABELS = {
  credit_score:        'Credit Score',
  debt_to_income:      'Debt-to-Income',
  loan_income_ratio:   'Loan-to-Income Ratio',
  monthly_installment: 'Monthly Installment',
  existing_loans:      'Existing Loans',
  income:              'Annual Income',
  coapplicant_income:  'Co-applicant Income',
  total_income:        'Total Income',
  age:                 'Age',
  employment_yrs:      'Employment Years',
  num_dependents:      'Dependents',
  education_grad:      'Education',
  self_employed:       'Employment Type',
  area_urban:          'Urban Area',
  area_semi_urban:     'Semi-Urban Area',
  credit_bucket:       'Credit Band',
  income_per_dependent:'Income per Dependent',
  loan_amount:         'Loan Amount',
  loan_term:           'Loan Term',
};

// ── Credit score band hint ─────────────────────────────────────────────────
function creditBand(score) {
  if (!score || score < 300 || score > 850) return '';
  if (score <= 580) return '⬇ Poor';
  if (score <= 670) return '◆ Fair';
  if (score <= 740) return '◆ Good';
  if (score <= 800) return '▲ Very Good';
  return '★ Exceptional';
}

// ── Toggle helpers ─────────────────────────────────────────────────────────
function setupToggle(groupId, hiddenId) {
  const group = document.getElementById(groupId);
  const hidden = document.getElementById(hiddenId);
  group.querySelectorAll('.toggle').forEach(btn => {
    btn.addEventListener('click', () => {
      group.querySelectorAll('.toggle').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      hidden.value = btn.dataset.val;
    });
  });
}

function setupAreaToggle() {
  const buttons = document.querySelectorAll('.area-toggle');
  const urbanInput    = document.getElementById('area_urban_val');
  const semiInput     = document.getElementById('area_semi_urban_val');

  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      buttons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const area = btn.dataset.area;
      urbanInput.value = area === 'urban' ? '1' : '0';
      semiInput.value  = area === 'semi'  ? '1' : '0';
    });
  });
}

// ── Load demo values ───────────────────────────────────────────────────────
function loadDemo() {
  document.getElementById('age').value             = DEMO_DATA.age;
  document.getElementById('income').value          = DEMO_DATA.income;
  document.getElementById('coapplicant_income').value = DEMO_DATA.coapplicant_income;
  document.getElementById('loan_amount').value     = DEMO_DATA.loan_amount;
  document.getElementById('loan_term').value       = DEMO_DATA.loan_term;
  document.getElementById('credit_score').value    = DEMO_DATA.credit_score;
  document.getElementById('employment_yrs').value  = DEMO_DATA.employment_yrs;
  document.getElementById('existing_loans').value  = DEMO_DATA.existing_loans;
  document.getElementById('num_dependents').value  = DEMO_DATA.num_dependents;

  // education toggle
  const eduGroup = document.getElementById('education_grad');
  eduGroup.querySelectorAll('.toggle').forEach(b => {
    b.classList.toggle('active', b.dataset.val === String(DEMO_DATA.education_grad));
  });
  document.getElementById('education_grad_val').value = DEMO_DATA.education_grad;

  // employment toggle
  const empGroup = document.getElementById('self_employed');
  empGroup.querySelectorAll('.toggle').forEach(b => {
    b.classList.toggle('active', b.dataset.val === String(DEMO_DATA.self_employed));
  });
  document.getElementById('self_employed_val').value = DEMO_DATA.self_employed;

  // area toggle
  document.querySelectorAll('.area-toggle').forEach(b => {
    b.classList.toggle('active', b.dataset.area === DEMO_DATA.area);
  });
  document.getElementById('area_urban_val').value     = DEMO_DATA.area === 'urban' ? '1' : '0';
  document.getElementById('area_semi_urban_val').value = DEMO_DATA.area === 'semi'  ? '1' : '0';

  // trigger credit hint
  document.getElementById('credit-band').textContent = creditBand(DEMO_DATA.credit_score);
}

// ── Build payload from form ────────────────────────────────────────────────
function buildPayload() {
  return {
    age:                parseInt(document.getElementById('age').value),
    income:             parseFloat(document.getElementById('income').value),
    coapplicant_income: parseFloat(document.getElementById('coapplicant_income').value),
    loan_amount:        parseFloat(document.getElementById('loan_amount').value),
    loan_term:          parseInt(document.getElementById('loan_term').value),
    credit_score:       parseInt(document.getElementById('credit_score').value),
    employment_yrs:     parseInt(document.getElementById('employment_yrs').value),
    existing_loans:     parseInt(document.getElementById('existing_loans').value),
    num_dependents:     parseInt(document.getElementById('num_dependents').value),
    education_grad:     parseInt(document.getElementById('education_grad_val').value),
    self_employed:      parseInt(document.getElementById('self_employed_val').value),
    area_urban:         parseInt(document.getElementById('area_urban_val').value),
    area_semi_urban:    parseInt(document.getElementById('area_semi_urban_val').value),
  };
}

// ── Render result ──────────────────────────────────────────────────────────
function renderResult(data) {
  const approved = data.decision === 'Approved';
  const prob     = data.probability_approved;

  // Show/hide panels
  document.getElementById('result-empty').hidden   = true;
  document.getElementById('result-error').hidden   = true;
  document.getElementById('result-content').hidden = false;

  // Verdict badge
  const badge = document.getElementById('verdict-badge');
  badge.className = `verdict-badge ${approved ? 'approved' : 'rejected'}`;
  badge.textContent = approved ? '✓' : '✕';

  // Verdict text
  const vtext = document.getElementById('verdict-text');
  vtext.className = `verdict-text ${approved ? 'approved' : 'rejected'}`;
  vtext.textContent = data.decision;

  // Probability bar (animate after a tick)
  const fill = document.getElementById('prob-fill');
  fill.className = `prob-fill ${approved ? 'approved' : 'rejected'}`;
  fill.style.width = '0';
  document.getElementById('prob-pct').textContent = `${Math.round(prob * 100)}%`;
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      fill.style.width = `${prob * 100}%`;
    });
  });

  // Summary strip
  const s = data.input_summary;
  document.getElementById('sum-credit').textContent  = s.credit_score;
  document.getElementById('sum-income').textContent  = formatINR(s.total_income);
  document.getElementById('sum-loan').textContent    = formatINR(s.loan_amount);
  document.getElementById('sum-dti').textContent     = s.debt_to_income.toFixed(2);

  // SHAP factors
  renderFactors('factors-for',     data.explanation.supporting_factors, 'pos');
  renderFactors('factors-against', data.explanation.opposing_factors,   'neg');
}

function renderFactors(listId, factors, sign) {
  const ul = document.getElementById(listId);
  ul.innerHTML = '';

  if (!factors || factors.length === 0) {
    ul.innerHTML = `<li class="factors-col--empty">None</li>`;
    return;
  }

  factors.forEach(f => {
    const li = document.createElement('li');
    li.className = 'factor-item';
    const label   = FEATURE_LABELS[f.feature] || f.feature;
    const impact  = f.impact;
    li.innerHTML = `
      <span class="factor-name">${label}</span>
      <span class="factor-impact ${sign}">${impact > 0 ? '+' : ''}${impact.toFixed(4)}</span>
    `;
    ul.appendChild(li);
  });
}

// ── Helpers ────────────────────────────────────────────────────────────────
function formatINR(val) {
  if (val >= 1_00_000) return `₹${(val / 1_00_000).toFixed(1)}L`;
  if (val >= 1000)     return `₹${(val / 1000).toFixed(0)}k`;
  return `₹${val}`;
}

function setLoading(yes) {
  const btn    = document.getElementById('submit-btn');
  const label  = btn.querySelector('.btn-text');
  const spin   = btn.querySelector('.btn-spinner');
  btn.disabled = yes;
  label.textContent = yes ? 'Evaluating…' : 'Evaluate Application';
  spin.hidden = !yes;
}

function showError(msg) {
  document.getElementById('result-empty').hidden   = true;
  document.getElementById('result-content').hidden = true;
  document.getElementById('result-error').hidden   = false;
  document.getElementById('error-msg').textContent = msg || 'Something went wrong.';
}

// ── Form submit ────────────────────────────────────────────────────────────
async function handleSubmit(e) {
  e.preventDefault();
  setLoading(true);

  try {
    const payload = buildPayload();

    const res = await fetch(`${API_BASE}/predict`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });

    const data = await res.json();

    if (!res.ok) {
      showError(data.error || `Server error ${res.status}`);
    } else {
      renderResult(data);
    }
  } catch (err) {
    showError('Could not reach the API. Is the server running on port 5002?');
  } finally {
    setLoading(false);
  }
}

// ── Reset ──────────────────────────────────────────────────────────────────
function resetResult() {
  document.getElementById('result-content').hidden = true;
  document.getElementById('result-error').hidden   = true;
  document.getElementById('result-empty').hidden   = false;
}

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  setupToggle('education_grad', 'education_grad_val');
  setupToggle('self_employed',  'self_employed_val');
  setupAreaToggle();

  // Credit score live hint
  document.getElementById('credit_score').addEventListener('input', e => {
    document.getElementById('credit-band').textContent = creditBand(Number(e.target.value));
  });

  document.getElementById('predict-form').addEventListener('submit', handleSubmit);
  document.getElementById('demo-btn').addEventListener('click', loadDemo);
  document.getElementById('reset-btn').addEventListener('click', resetResult);
  document.getElementById('retry-btn').addEventListener('click', resetResult);
});
