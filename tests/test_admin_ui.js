// SPDX-License-Identifier: Apache-2.0
// Dashboard contracts run without a browser, network, or model inference.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const admin = process.argv[2];
const source = fs.readFileSync(path.join(admin, 'static/js/dashboard.js'), 'utf8');
const strings = JSON.parse(fs.readFileSync(path.join(admin, 'i18n/en.json'), 'utf8'));
const requests = [];
const storage = new Map([['omlx_preset_cache', JSON.stringify({ presets: [{ name: 'remote' }] })]]);
const media = { matches: false, addEventListener() {}, removeEventListener() {} };
const window = {
    location: { href: 'http://localhost/admin/dashboard?tab=cluster', search: '?tab=cluster' },
    history: { replaceState(_a, _b, url) { window.location.href = String(url); } },
    t: key => strings[key],
    matchMedia: () => media,
};
const context = vm.createContext({
    window, URL, URLSearchParams, console, Date,
    localStorage: { getItem: key => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value) },
    document: { documentElement: { setAttribute(key, value) { this[key] = value; } } },
    setTimeout: () => 1, clearTimeout() {}, setInterval: () => 1, clearInterval() {},
    EventSource: class { constructor() { this.closed = false; } close() { this.closed = true; } },
    fetch: async (url, options) => {
        requests.push({ url, options });
        return { ok: true, status: 200, json: async () => ({ presets: [{ name: 'bundled' }] }) };
    },
});
vm.runInContext(source, context);
const make = () => vm.runInContext('dashboard()', context);
const response = (status, data = {}) => ({ ok: status < 400, status, json: async () => data, clone() { return this; } });

(async () => {
    const app = make();
    app.applyTabStateFromUrl();
    assert.equal(app.mainTab, 'status');
    app.setModelsTab('uploader');
    assert.equal(app.mainTab, 'models');
    assert.equal(app.modelsTab, 'uploader');
    assert.match(window.location.href, /modelsTab=uploader/);
    app.setMainTab('cluster');
    assert.equal(app.mainTab, 'models');
    requests.length = 0;
    await app.loadPresets();
    assert.equal(requests.length, 1);
    assert.equal(requests[0].url, '/admin/static/omlx_preset.json');
    assert.equal(app.presets[0].name, 'bundled');

    app.selectedStatsModel = '';
    app.updateRecentThroughput({ uptime_seconds: 10, total_completion_tokens: 100 });
    assert.equal(app.recentGenerationTPS, null);
    app.updateRecentThroughput({ uptime_seconds: 20, total_completion_tokens: 300 });
    assert.equal(app.recentGenerationTPS, 20);
    app.updateRecentThroughput({ uptime_seconds: 81, total_completion_tokens: 300 });
    assert.equal(app.recentGenerationTPS, null);
    app.updateRecentThroughput({ uptime_seconds: 82, total_completion_tokens: 300 });
    assert.equal(app.recentGenerationTPS, 0);
    app.selectedStatsModel = 'other';
    app.updateRecentThroughput({ uptime_seconds: 83, total_completion_tokens: 2000 });
    assert.equal(app.recentGenerationTPS, null);
    app.updateRecentThroughput({ uptime_seconds: 1, total_completion_tokens: 0 });
    assert.equal(app.recentGenerationTPS, null);

    app.loadModels = async () => {};
    app.benchRunning = true;
    app.connectBenchSSE('fixture');
    const stream = app.benchEventSource;
    const result = { type: 'result', data: { test_type: 'single', pp: 1024, tg: 128, gen_tps: 42 } };
    stream.onmessage({ data: JSON.stringify(result) });
    stream.onmessage({ data: JSON.stringify(result) });
    assert.equal(app.benchSingleResults.length, 1);
    stream.onmessage({ data: JSON.stringify({ type: 'done' }) });
    assert.equal(app.benchRunning, false);
    assert.equal(stream.closed, true);
    assert.equal(app.benchEventSource, null);
    assert.equal(app.notice.message, strings['shell.benchmark_done']);

    app.showHfMirrorModal = true;
    app.hfMirrorEndpoint = 'https://mirror.example';
    context.fetch = async () => response(200);
    await app.saveHfMirrorEndpoint();
    assert.equal(app.showHfMirrorModal, false);
    assert.equal(app.globalSettings.huggingface.endpoint, 'https://mirror.example');
    assert.equal(app.notice.message, strings['js.success.settings_saved']);
    app.showHfMirrorModal = true;
    context.fetch = async () => { throw new Error('offline'); };
    await app.saveHfMirrorEndpoint();
    assert.equal(app.showHfMirrorModal, true);
    assert.equal(app.notice.error, true);
    assert.equal(app.hfMirrorSaving, false);

    app.showClearStatsConfirm = true;
    app.loadStats = async () => {};
    context.fetch = async () => response(500);
    await app.clearStats();
    assert.equal(app.showClearStatsConfirm, true);
    assert.equal(app.notice.error, true);
    context.fetch = async () => response(200);
    await app.clearStats();
    assert.equal(app.showClearStatsConfirm, false);
    assert.equal(app.notice.message, strings['shell.cleared']);
    context.fetch = async () => response(500, { detail: 'Task cannot be cancelled' });
    await app.actionRequest('/fixture', { method: 'POST' });
    assert.equal(app.notice.message, 'Task cannot be cancelled');
    assert.equal(app.notice.error, true);

    context.navigator = { clipboard: { writeText: async () => { throw new Error('denied'); } } };
    window.isSecureContext = true;
    app._copyFallback = () => false;
    assert.equal(await app.copyToClipboard('fixture'), false);
    assert.equal(app.notice.error, true);
    assert.equal(app.notice.message, strings['shell.copy_failed']);
    context.navigator.clipboard.writeText = async () => {};
    assert.equal(await app.copyToClipboard('fixture'), true);
    assert.equal(app.notice.message, strings['login.setup.copied']);

    app.stats.cli_prefix = 'omlx';
    assert.equal(app.claudeCommand, 'fmlx launch claude');
    app.setTheme('dark');
    assert.equal(context.document.documentElement['data-theme'], 'dark');
    assert.equal(storage.get('omlx-chat-theme'), 'dark');
    console.log('Dashboard behavior contracts passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
