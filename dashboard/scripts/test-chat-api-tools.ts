/**
 * Test the main Chat API endpoint with tools
 */

const DASHBOARD_URL = 'http://localhost:3000';
const WORKER_URL = 'https://e2e-testing-agent.samuelvinay-kumar.workers.dev';

interface TestResult {
  name: string;
  passed: boolean;
  error?: string;
  data?: unknown;
}

const results: TestResult[] = [];

function log(message: string) {
  console.log(`\n${'='.repeat(60)}\n${message}\n${'='.repeat(60)}`);
}

function success(name: string, data?: unknown) {
  console.log(`✅ PASS: ${name}`);
  results.push({ name, passed: true, data });
}

function fail(name: string, error: string) {
  console.log(`❌ FAIL: ${name} - ${error}`);
  results.push({ name, passed: false, error });
}

async function testWorkerObserve(): Promise<void> {
  log('TEST 1: Worker /observe endpoint');

  try {
    const response = await fetch(`${WORKER_URL}/observe`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url: 'https://demo.vercel.store',
        instruction: 'What actions can I take on this page?',
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }

    const data = await response.json();
    console.log('Response:', JSON.stringify(data, null, 2).substring(0, 500) + '...');

    if (data.actions && Array.isArray(data.actions)) {
      console.log(`Found ${data.actions.length} interactive elements`);
      success('Worker /observe endpoint', { actionsCount: data.actions.length });
    } else {
      throw new Error('No actions array in response');
    }
  } catch (err) {
    fail('Worker /observe endpoint', (err as Error).message);
  }
}

async function testWorkerExtract(): Promise<void> {
  log('TEST 2: Worker /extract endpoint');

  try {
    const response = await fetch(`${WORKER_URL}/extract`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url: 'https://demo.vercel.store',
        instruction: 'Extract all product names from this page',
        schema: { products: 'array of product name strings' },
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }

    const data = await response.json();
    console.log('Response:', JSON.stringify(data, null, 2).substring(0, 500) + '...');

    success('Worker /extract endpoint', data);
  } catch (err) {
    fail('Worker /extract endpoint', (err as Error).message);
  }
}

async function testWorkerAct(): Promise<void> {
  log('TEST 3: Worker /act endpoint');

  try {
    const response = await fetch(`${WORKER_URL}/act`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url: 'https://demo.vercel.store',
        instruction: 'Click on the first product',
        selfHeal: true,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }

    const data = await response.json();
    console.log('Response:', JSON.stringify(data, null, 2).substring(0, 500) + '...');

    success('Worker /act endpoint', data);
  } catch (err) {
    fail('Worker /act endpoint', (err as Error).message);
  }
}

async function testWorkerTest(): Promise<void> {
  log('TEST 4: Worker /test endpoint');

  try {
    const response = await fetch(`${WORKER_URL}/test`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url: 'https://demo.vercel.store',
        steps: [
          'Click on any product card',
          'Verify product page loads',
        ],
        browser: 'chrome',
        screenshot: false,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }

    const data = await response.json();
    console.log('Response:', JSON.stringify(data, null, 2).substring(0, 1000) + '...');

    success('Worker /test endpoint', data);
  } catch (err) {
    fail('Worker /test endpoint', (err as Error).message);
  }
}

async function testWorkerAgent(): Promise<void> {
  log('TEST 5: Worker /agent endpoint');

  try {
    const response = await fetch(`${WORKER_URL}/agent`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url: 'https://demo.vercel.store',
        instruction: 'Navigate to the search page and look for products',
        maxSteps: 3,
        captureScreenshots: false,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }

    const data = await response.json();
    console.log('Response:', JSON.stringify(data, null, 2).substring(0, 1000) + '...');

    success('Worker /agent endpoint', data);
  } catch (err) {
    fail('Worker /agent endpoint', (err as Error).message);
  }
}

async function testWorkerHealth(): Promise<void> {
  log('TEST 6: Worker health check');

  try {
    const response = await fetch(`${WORKER_URL}/health`);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }

    const data = await response.json();
    console.log('Response:', JSON.stringify(data, null, 2));

    success('Worker health check', data);
  } catch (err) {
    fail('Worker health check', (err as Error).message);
  }
}

async function printSummary(): Promise<void> {
  log('TEST SUMMARY');

  const passed = results.filter((r) => r.passed).length;
  const failed = results.filter((r) => !r.passed).length;

  console.log(`\nTotal Tests: ${results.length}`);
  console.log(`Passed: ${passed}`);
  console.log(`Failed: ${failed}`);
  console.log(`\n${'='.repeat(60)}`);

  if (failed > 0) {
    console.log('\nFailed Tests:');
    results.filter((r) => !r.passed).forEach((r) => {
      console.log(`  - ${r.name}: ${r.error}`);
    });
  }

  console.log(`\n${'='.repeat(60)}`);
  console.log(failed === 0 ? '\n🎉 ALL TESTS PASSED!' : '\n⚠️ SOME TESTS FAILED');
}

async function runTests(): Promise<void> {
  console.log('\n🧪 Starting Cloudflare Worker API Tests\n');
  console.log(`Worker URL: ${WORKER_URL}`);

  // Test 6: Worker health check (quick check first)
  await testWorkerHealth();

  // Test 1: Worker /observe endpoint
  await testWorkerObserve();

  // Test 2: Worker /extract endpoint
  await testWorkerExtract();

  // Test 3: Worker /act endpoint
  await testWorkerAct();

  // Test 4: Worker /test endpoint
  await testWorkerTest();

  // Test 5: Worker /agent endpoint
  await testWorkerAgent();

  // Summary
  await printSummary();
}

// Run tests
runTests().catch(console.error);
