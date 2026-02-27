const express = require('express');
const registry = require('../bugs/registry');

const router = express.Router();

/**
 * GET /chaos/bugs — list all bugs with their current state
 */
router.get('/bugs', (req, res) => {
  const bugs = registry.listBugs();

  // Optional category filter
  if (req.query.category) {
    const filtered = bugs.filter((b) => b.category === req.query.category);
    return res.json({
      bugs: filtered,
      total: filtered.length,
      activeCount: filtered.filter((b) => b.enabled).length,
    });
  }

  res.json({
    bugs,
    total: bugs.length,
    activeCount: registry.getActiveBugCount(),
    scenarios: Object.keys(registry.SCENARIOS),
  });
});

/**
 * POST /chaos/bugs/:id/enable — enable a specific bug
 */
router.post('/bugs/:id/enable', (req, res) => {
  const bug = registry.getBug(req.params.id);
  if (!bug) {
    return res.status(404).json({ error: `Bug '${req.params.id}' not found` });
  }

  registry.enableBug(req.params.id);

  res.json({
    id: req.params.id,
    ...bug,
    message: `Bug '${req.params.id}' enabled`,
  });
});

/**
 * POST /chaos/bugs/:id/disable — disable a specific bug
 */
router.post('/bugs/:id/disable', (req, res) => {
  const bug = registry.getBug(req.params.id);
  if (!bug) {
    return res.status(404).json({ error: `Bug '${req.params.id}' not found` });
  }

  registry.disableBug(req.params.id);

  res.json({
    id: req.params.id,
    ...bug,
    message: `Bug '${req.params.id}' disabled`,
  });
});

/**
 * POST /chaos/bugs/reset — disable all bugs
 */
router.post('/bugs/reset', (req, res) => {
  registry.resetAll();

  res.json({
    message: 'All bugs disabled',
    activeCount: 0,
  });
});

/**
 * POST /chaos/bugs/scenario/:name — enable a preset scenario
 */
router.post('/bugs/scenario/:name', (req, res) => {
  const scenarioName = req.params.name;
  const scenario = registry.SCENARIOS[scenarioName];

  if (!scenario) {
    return res.status(404).json({
      error: `Scenario '${scenarioName}' not found`,
      available: Object.keys(registry.SCENARIOS),
    });
  }

  // Optionally reset first
  if (req.query.reset !== 'false') {
    registry.resetAll();
  }

  registry.enableScenario(scenarioName);

  res.json({
    scenario: scenarioName,
    enabledBugs: scenario,
    activeCount: registry.getActiveBugCount(),
    message: `Scenario '${scenarioName}' activated with ${scenario.length} bugs`,
  });
});

/**
 * GET /chaos/health — chaos system health + active bug count
 */
router.get('/health', (req, res) => {
  const activeBugs = registry.listBugs().filter((b) => b.enabled);

  res.json({
    status: 'ok',
    service: 'chaos-controller',
    activeBugCount: activeBugs.length,
    activeBugs: activeBugs.map((b) => b.id),
    totalBugs: registry.listBugs().length,
    scenarios: Object.keys(registry.SCENARIOS),
  });
});

module.exports = router;
