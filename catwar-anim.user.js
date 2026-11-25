// ==UserScript==
// @name         Живые локации CatWar
// @namespace    https://github.com/Qkinawa/cw-bg-anim
// @version      1.0.202611250554
// @description  Анимированные фоны локаций
// @author       Qkinawa
// @match        https://catwar.su/cw3*
// @match        https://catwar.net/cw3*
// @noframes
// @run-at       document-idle
// @grant        GM_getValue
// @grant        GM_setValue
// @grant        GM_addValueChangeListener
// @grant        GM_registerMenuCommand
// @grant        GM_unregisterMenuCommand
// @updateURL    https://cw.qkinawa.com/catwar-anim.user.js
// @downloadURL  https://cw.qkinawa.com/catwar-anim.user.js
// ==/UserScript==

(function () {
  'use strict';

  const BASE = 'https://cw.qkinawa.com/';

  const BG = {
    '80005': '2129ee6e', '80013': '8fa96395', '84000': 'dc53ad3a', '84001': '611fa36f',
    '84002': '489c65b0', '84003': 'f46db8ec', '84004': 'c7ec0664', '84005': '2fac3653',
    '84006': 'bf95273c', '84007': '8354fb53', '84008': 'f7e34348', '84009': '674b0c5d',
    '84010': '209e00de', '84011': 'ffc48472', '84012': '05498a62', '84013': 'af9f8651',
    '84014': 'c44f9b28', '84015': 'c2f5500e', '84016': 'fa74115f', '84017': 'ae8c2ff8',
    '84018': '1349948e', '84019': '5623d84f', '84020': '044e9ad1', '84021': '85208869',
    '84022': 'bdb1e6c7', '84023': '859b8727', '84024': '4e3c48d9', '84025': 'fc7aa379',
    '84026': 'ed0c03b4', '84027': '2d5e960e', '84029': '99a15e2e', '84030': 'abda772e',
    '84031': '631a57ef', '84032': '7dc3efd0', '84033': 'e4d40459', '84034': '9ad06343',
    '84035': 'b2077e30', '84036': '9a2d351f', '84037': 'b9be8f4f', '84038': '20e7a07a',
    '84039': '931a0b44', '84040': '0751b703', '84041': 'c5b3dce2', '84042': '463eb9a4',
    '84043': 'ae85472c', '84044': '8852c440', '84045': '44425977', '84046': '754c92dc',
    '84047': '3eb98134', '84048': '67a614e0', '84049': 'e415eec5', '84050': '747d10d4',
    '84051': '1ee17c2f', '84052': '587cf9e2', '84053': '44b2350b', '84054': '78f5f543',
    '84055': '2041a0fc', '84056': '046dfebe', '84057': '34db80fe', '84058': '9505627f',
    '84059': '5b66c862', '84060': '15ccdb63', '84061': 'c75df071', '84062': '93bfac58',
    '84063': 'e8b63301', '84064': '85498f66', '84065': '33e7179a', '84066': '9d3dd46b',
    '84067': 'a837664e', '84068': '2aef0b66', '84069': 'da00fc63', '84070': 'a783f698',
    '84073': 'ec8673aa', '84074': 'ed61f2bc', '84075': '4a9d713c', '84076': 'cf2720fd',
    '84077': '9b4ae096', '84078': '5d6641f1', '84079': '47b9077a', '84080': '1ee41255',
    '84081': '2ca96d59', '84082': '3bf57898', '84083': '43e08746', '84084': '5e463e24',
    '84085': 'a3b06903', '84086': '2dda1ea4', '84087': '3e9a91a1', '84088': '78853fcb',
    '84089': 'b4bfcda5', '84090': 'a84126fb', '84100': 'd0720dba', '84101': 'f94256b8',
    '84102': 'd16c8233', '84103': '60263076', '84104': 'c0624780', '84870': '6bc82eb6',
    '84894': '67cbac9f', '84895': 'f6cc9d00', '84896': 'ffaea824', '84897': 'c810773d',
  };

  const canStore = typeof GM_getValue === 'function' && typeof GM_setValue === 'function';
  const canMenu = typeof GM_registerMenuCommand === 'function';
  const canUnregister = typeof GM_unregisterMenuCommand === 'function';
  const own = (obj, key) => Object.prototype.hasOwnProperty.call(obj, key);

  function getSetting(name, fallback) {
    return canStore ? GM_getValue(name, fallback) : fallback;
  }

  function setSetting(name, value) {
    if (canStore) GM_setValue(name, value);
  }

  function readOff() {
    const value = getSetting('off', []);
    return new Set(Array.isArray(value) ? value : []);
  }

  let off = readOff();
  let allOff = Boolean(getSetting('allOff', false));

  const ready = new Set();
  const broken = new Map();
  let pending = null;
  let shown = null;
  let queued = false;
  let lastBg = null;
  let lastKey = null;
  let menu = [];
  let menuState = null;

  const style = document.createElement('style');
  (document.head || document.documentElement).appendChild(style);

  function fileName(key) {
    return key.replace(/\.\w+$/, '').replace(/\W/g, '_') + '.webp';
  }

  function urlOf(key) {
    return BASE + 'bg/' + fileName(key) + '?v=' + BG[key];
  }

  function keyOf(bg) {
    const m = bg.match(/url\(["']?([^"')]+)["']?\)/);
    if (!m) return null;
    const path = m[1].split('?')[0].replace(/^https?:\/\/[^/]+/, '').replace(/^(?:\.?\/)?(?:cw3\/)?/, '');
    if (own(BG, path)) return path;
    const id = (path.match(/(\d+)\.\w+$/) || [])[1];
    return id && own(BG, id) ? id : null;
  }

  function show(key) {
    if (shown === key) return;
    shown = key;
    style.textContent = key ? `#cages_div { background-image: url("${urlOf(key)}") !important; }` : '';
  }

  function cancel() {
    if (!pending) return;
    pending.img.onload = pending.img.onerror = null;
    pending.img.src = '';
    pending = null;
  }

  function preload(key) {
    if (pending && pending.key === key) return;
    cancel();
    const img = new Image();
    pending = { key, img };
    img.onload = () => {
      const done = () => {
        if (pending && pending.img === img) pending = null;
        ready.add(key);
        update();
      };
      if (img.decode) img.decode().then(done, done);
      else done();
    };
    img.onerror = () => {
      if (pending && pending.img === img) pending = null;
      broken.set(key, Date.now());
    };
    img.src = urlOf(key);
  }

  function update() {
    queued = false;
    const div = document.getElementById('cages_div');
    const bg = div ? div.style.backgroundImage : '';
    if (bg !== lastBg) {
      lastBg = bg;
      lastKey = bg ? keyOf(bg) : null;
    }
    const key = lastKey;
    updateMenu(key);

    if (!key || allOff || off.has(key) || Date.now() - (broken.get(key) || 0) < 60000) {
      cancel();
      show(null);
      return;
    }
    if (ready.has(key)) {
      cancel();
      show(key);
      return;
    }
    show(null);
    preload(key);
  }

  function schedule() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(update);
  }

  function toggle(key) {
    if (!key) return;
    off = readOff();
    if (off.has(key)) off.delete(key);
    else off.add(key);
    setSetting('off', [...off]);
    update();
  }

  function enableLocations() {
    off = new Set();
    setSetting('off', []);
    update();
  }

  function setAllOff(value) {
    allOff = value;
    setSetting('allOff', value);
    update();
  }

  function updateMenu(key) {
    if (!canMenu || !canUnregister) return;
    const state = [key, allOff, key && off.has(key), off.size].join('|');
    if (state === menuState) return;
    menuState = state;
    menu.forEach((id) => GM_unregisterMenuCommand(id));
    menu = [];
    const add = (label, fn) => menu.push(GM_registerMenuCommand(label, fn));

    if (allOff) {
      add('Включить анимацию везде', () => setAllOff(false));
      return;
    }
    if (key) {
      add(off.has(key) ? 'Включить анимацию на этой локации' : 'Выключить анимацию на этой локации', () => toggle(key));
    }
    add('Выключить анимацию везде', () => setAllOff(true));
    if (off.size) {
      add(`Включить выключенные локации (${off.size})`, enableLocations);
    }
  }

  if (canMenu && !canUnregister) {
    GM_registerMenuCommand('Выключить/включить анимацию на этой локации', () => toggle(lastKey));
    GM_registerMenuCommand('Выключить/включить анимацию везде', () => setAllOff(!allOff));
  }

  if (typeof GM_addValueChangeListener === 'function') {
    GM_addValueChangeListener('off', (name, oldValue, newValue, remote) => {
      if (!remote) return;
      off = new Set(Array.isArray(newValue) ? newValue : []);
      update();
    });
    GM_addValueChangeListener('allOff', (name, oldValue, newValue, remote) => {
      if (!remote) return;
      allOff = Boolean(newValue);
      update();
    });
  } else {
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) return;
      off = readOff();
      allOff = Boolean(getSetting('allOff', false));
      update();
    });
  }

  new MutationObserver(schedule).observe(document.documentElement, {
    subtree: true,
    childList: true,
    attributes: true,
    attributeFilter: ['style'],
  });

  update();
})();
