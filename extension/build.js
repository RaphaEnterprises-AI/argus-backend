#!/usr/bin/env node

/**
 * Build script for Chrome Web Store packaging.
 *
 * Copies extension files to dist/ and validates the manifest.
 * Run `npm run package` to produce the final .zip.
 */

const fs = require('fs');
const path = require('path');

const SRC = __dirname;
const DIST = path.join(SRC, 'dist');

// Files to include in the distribution
const FILES = [
  'manifest.json',
  'background.js',
  'content.js',
  'recorder.js',
  'popup.html',
  'popup.js',
  'options.html',
  'options.js',
];

const DIRS = ['icons'];

// Clean dist
if (fs.existsSync(DIST)) {
  fs.rmSync(DIST, { recursive: true });
}
fs.mkdirSync(DIST, { recursive: true });

// Copy files
for (const file of FILES) {
  const src = path.join(SRC, file);
  const dest = path.join(DIST, file);
  if (!fs.existsSync(src)) {
    console.warn(`Warning: ${file} not found, skipping`);
    continue;
  }
  fs.copyFileSync(src, dest);
  console.log(`  Copied ${file}`);
}

// Copy directories
for (const dir of DIRS) {
  const srcDir = path.join(SRC, dir);
  const destDir = path.join(DIST, dir);
  if (!fs.existsSync(srcDir)) {
    console.warn(`Warning: ${dir}/ not found, skipping`);
    continue;
  }
  fs.mkdirSync(destDir, { recursive: true });
  for (const file of fs.readdirSync(srcDir)) {
    fs.copyFileSync(path.join(srcDir, file), path.join(destDir, file));
  }
  console.log(`  Copied ${dir}/`);
}

// Validate manifest
const manifest = JSON.parse(fs.readFileSync(path.join(DIST, 'manifest.json'), 'utf8'));
const requiredKeys = ['manifest_version', 'name', 'version', 'permissions', 'background'];
for (const key of requiredKeys) {
  if (!(key in manifest)) {
    console.error(`Error: manifest.json missing required key: ${key}`);
    process.exit(1);
  }
}

// Verify referenced files exist in dist
const referencedFiles = [
  manifest.background?.service_worker,
  manifest.action?.default_popup,
  manifest.options_page,
  ...(manifest.content_scripts || []).flatMap(cs => cs.js || []),
].filter(Boolean);

for (const ref of referencedFiles) {
  if (!fs.existsSync(path.join(DIST, ref))) {
    console.error(`Error: manifest references missing file: ${ref}`);
    process.exit(1);
  }
}

// Verify icon files
const icons = { ...manifest.icons, ...(manifest.action?.default_icon || {}) };
for (const [size, iconPath] of Object.entries(icons)) {
  if (!fs.existsSync(path.join(DIST, iconPath))) {
    console.error(`Error: icon file missing: ${iconPath} (${size}px)`);
    process.exit(1);
  }
}

console.log(`\nBuild complete: ${DIST}`);
console.log(`  Manifest: v${manifest.manifest_version}, ${manifest.name} ${manifest.version}`);
console.log(`  Files: ${FILES.length + DIRS.length} entries`);
console.log(`\nRun "npm run package" to create the .zip file.`);
