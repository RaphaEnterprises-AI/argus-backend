/**
 * E2E Testing Agent - rrweb Session Recorder
 *
 * This content script module handles session recording using rrweb.
 * It injects the rrweb library, captures DOM events, and uploads
 * recordings to the Skopaq backend for test conversion.
 *
 * Usage: Injected by background.js when user starts a recording.
 */

(function() {
  'use strict';

  // Avoid double-loading
  if (window.__argusRecorderLoaded) return;
  window.__argusRecorderLoaded = true;

  let recordingEvents = [];
  let stopFn = null;
  let isRecording = false;
  let startTime = null;
  let maxDuration = 300000; // 5 min default

  /**
   * Load rrweb library from CDN
   */
  function loadRrweb() {
    return new Promise((resolve, reject) => {
      if (window.rrweb) {
        resolve();
        return;
      }
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/rrweb@2.0.0-alpha.11/dist/rrweb.min.js';
      script.onload = () => resolve();
      script.onerror = () => reject(new Error('Failed to load rrweb'));
      document.head.appendChild(script);
    });
  }

  /**
   * Start recording the current page
   */
  async function startRecording(options = {}) {
    if (isRecording) {
      return { success: false, error: 'Already recording' };
    }

    try {
      await loadRrweb();

      recordingEvents = [];
      startTime = Date.now();
      maxDuration = options.maxDuration || 300000;

      stopFn = window.rrweb.record({
        emit(event) {
          recordingEvents.push(event);
        },
        maskAllInputs: options.maskInputs !== false,
        recordCanvas: options.recordCanvas || false,
        sampling: {
          mousemove: 50,
          mouseInteraction: true,
          scroll: 150,
          input: 'last',
        },
      });

      isRecording = true;

      // Auto-stop after max duration
      setTimeout(() => {
        if (isRecording) {
          stopRecording();
          // Notify background script
          chrome.runtime.sendMessage({
            type: 'recordingStopped',
            reason: 'maxDuration',
            eventCount: recordingEvents.length,
          });
        }
      }, maxDuration);

      return {
        success: true,
        data: {
          startTime,
          url: window.location.href,
          title: document.title,
        },
      };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  /**
   * Stop recording and return events
   */
  function stopRecording() {
    if (!isRecording) {
      return { success: false, error: 'Not recording' };
    }

    if (stopFn) {
      stopFn();
      stopFn = null;
    }

    isRecording = false;
    const duration = Date.now() - startTime;

    return {
      success: true,
      data: {
        events: recordingEvents,
        metadata: {
          url: window.location.href,
          title: document.title,
          startTime,
          duration,
          eventCount: recordingEvents.length,
          userAgent: navigator.userAgent,
          viewport: {
            width: window.innerWidth,
            height: window.innerHeight,
          },
        },
      },
    };
  }

  /**
   * Get recording status
   */
  function getRecordingStatus() {
    return {
      success: true,
      data: {
        isRecording,
        eventCount: recordingEvents.length,
        duration: isRecording ? Date.now() - startTime : 0,
        url: window.location.href,
      },
    };
  }

  // Listen for messages from background script
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === 'startRecording') {
      startRecording(message.options || {}).then(sendResponse);
      return true; // async
    }

    if (message.action === 'stopRecording') {
      sendResponse(stopRecording());
      return false;
    }

    if (message.action === 'getRecordingStatus') {
      sendResponse(getRecordingStatus());
      return false;
    }
  });
})();
