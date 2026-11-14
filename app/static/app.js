// Front Page client: feedback buttons, load more, reader Back, scroll
// restoration and the service worker. No framework, no build step.
(function () {
  "use strict";

  const feed = document.getElementById("feed");
  const status = document.getElementById("status");
  const FEEDBACK_KEY = "fp-feedback"; // article id -> latest button state
  const FEED_KEY_PREFIX = "fp-feed:"; // saved feed page, per URL
  const FEED_SNAPSHOT_MAX_AGE = 60 * 60 * 1000;

  // sessionStorage can be missing or throw (private mode, storage full);
  // everything here must keep working without it.
  function storageGet(key) {
    try {
      const raw = sessionStorage.getItem(key);
      return raw ? JSON.parse(raw) : null;
    } catch (e) {
      return null;
    }
  }
  function storageSet(key, value) {
    try {
      sessionStorage.setItem(key, JSON.stringify(value));
    } catch (e) {
      /* ignore */
    }
  }
  function storageRemove(key) {
    try {
      sessionStorage.removeItem(key);
    } catch (e) {
      /* ignore */
    }
  }

  function announce(text) {
    if (status) status.textContent = text;
  }

  // ---- feedback -----------------------------------------------------------

  function sendFeedback(articleId, kind, value) {
    return fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ article_id: articleId, kind: kind, value: value }),
    }).then(function (response) {
      if (!response.ok) throw new Error("HTTP " + response.status);
      return response.json();
    });
  }

  // holder: a card, or the reader's top bar; anything with data-id.
  function applyState(holder, state) {
    const pressed = { up: state.up, down: state.down, save: state.saved };
    holder.querySelectorAll("button[data-kind]").forEach(function (button) {
      button.setAttribute("aria-pressed", pressed[button.dataset.kind] ? "true" : "false");
    });
    if (holder.classList.contains("card")) holder.classList.toggle("is-dismissed", !!state.down);
  }

  function currentState(holder) {
    const state = {};
    holder.querySelectorAll("button[data-kind]").forEach(function (button) {
      state[button.dataset.kind === "save" ? "saved" : button.dataset.kind] =
        button.getAttribute("aria-pressed") === "true";
    });
    return state;
  }

  // Remember the newest state so the feed can update its card when the
  // reader is left with Back (the feed page may come from the bfcache).
  function rememberState(articleId, state) {
    const all = storageGet(FEEDBACK_KEY) || {};
    all[articleId] = { up: !!state.up, down: !!state.down, saved: !!state.saved };
    storageSet(FEEDBACK_KEY, all);
  }

  const MESSAGES = {
    up: ["Marked: more like this", "Removed like"],
    down: ["Marked: less like this", "Removed dislike"],
    save: ["Saved for later", "Removed from saved"],
  };

  function onFeedbackClick(button) {
    const holder = button.closest("[data-id]");
    const articleId = Number(holder.dataset.id);
    const kind = button.dataset.kind;
    const value = button.getAttribute("aria-pressed") !== "true";
    const before = currentState(holder);

    // Optimistic update so the tap feels instant; rolled back on failure.
    const next = Object.assign({}, before);
    next[kind === "save" ? "saved" : kind] = value;
    if (value && kind === "up") next.down = false;
    if (value && kind === "down") next.up = false;
    applyState(holder, next);
    button.disabled = true;

    sendFeedback(articleId, kind, value)
      .then(function (state) {
        applyState(holder, state);
        rememberState(articleId, state);
        announce(MESSAGES[kind][value ? 0 : 1]);
      })
      .catch(function () {
        applyState(holder, before);
        announce("Could not save your feedback. Check the connection and try again.");
      })
      .finally(function () {
        button.disabled = false;
      });
  }

  document.addEventListener("click", function (event) {
    const button = event.target.closest("button[data-kind]");
    if (button) {
      onFeedbackClick(button);
      return;
    }
    // Reader "Back": go back in history when we came from inside the app, so
    // the feed comes back exactly where it was (bfcache or our snapshot).
    const back = event.target.closest("a[data-back]");
    if (back && cameFromApp()) {
      event.preventDefault();
      history.back();
    }
  });

  function cameFromApp() {
    if (window.history.length < 2 || !document.referrer) return false;
    try {
      return new URL(document.referrer).origin === window.location.origin;
    } catch (e) {
      return false;
    }
  }

  // A broken or missing image is removed: on cards the typographic cover
  // underneath shows instead; in the reader the text simply closes up.
  document.addEventListener(
    "error",
    function (event) {
      const target = event.target;
      if (target.tagName !== "IMG") return;
      if (target.closest(".card-media") || target.closest(".prose")) target.remove();
      else if (target.closest(".reader-hero")) target.closest(".reader-hero").remove();
    },
    true
  );

  // ---- feed: load more ----------------------------------------------------

  const more = document.getElementById("load-more");
  if (more && feed) {
    more.addEventListener("click", function () {
      more.disabled = true;
      more.textContent = "Loading…";
      // Send the ids already shown rather than an offset: feedback reorders
      // the list, and this way nothing is repeated or skipped.
      const seen = Array.from(feed.querySelectorAll(".card")).map(function (card) {
        return card.dataset.id;
      });
      const params = new URLSearchParams({ tab: feed.dataset.tab, seen: seen.join(",") });
      fetch("/api/articles?" + params.toString())
        .then(function (response) {
          if (!response.ok) throw new Error("HTTP " + response.status);
          return response.json();
        })
        .then(function (data) {
          const holder = document.createElement("div");
          holder.innerHTML = data.html;
          let added = 0;
          let first = null;
          Array.from(holder.children).forEach(function (card) {
            if (feed.querySelector('.card[data-id="' + card.dataset.id + '"]')) return;
            feed.appendChild(card);
            first = first || card;
            added += 1;
          });
          more.hidden = !data.has_more;
          announce(added + " more articles loaded");
          if (first) first.querySelector(".card-link").focus({ preventScroll: true });
        })
        .catch(function () {
          announce("Could not load more articles.");
        })
        .finally(function () {
          more.disabled = false;
          more.textContent = "Load more";
        });
    });
  }

  // ---- feed: keep the place when coming back from the reader --------------
  //
  // Usually the browser restores the feed from its back/forward cache with
  // scroll position and "Load more" pages intact. When it reloads the page
  // instead, the server would render a freshly ranked list (opening an
  // article retrains the model), so the old scroll offset would point at
  // different articles. To avoid that we snapshot the feed when leaving and
  // put the very same cards back on a back/forward navigation.

  if (feed) {
    const feedKey = FEED_KEY_PREFIX + window.location.pathname + window.location.search;
    if ("scrollRestoration" in history) history.scrollRestoration = "manual";

    window.addEventListener("pagehide", function () {
      storageSet(feedKey, {
        html: feed.innerHTML,
        y: window.scrollY,
        moreHidden: more ? more.hidden : true,
        at: Date.now(),
      });
    });

    const nav = performance.getEntriesByType && performance.getEntriesByType("navigation")[0];
    const snapshot = storageGet(feedKey);
    if (nav && nav.type === "back_forward" && snapshot && Date.now() - snapshot.at < FEED_SNAPSHOT_MAX_AGE) {
      feed.innerHTML = snapshot.html;
      if (more) more.hidden = snapshot.moreHidden;
      const restore = function () {
        window.scrollTo(0, snapshot.y);
      };
      restore();
      requestAnimationFrame(restore);
      if (document.fonts && document.fonts.ready) document.fonts.ready.then(restore);
    }

    // Apply feedback given in the reader (runs on bfcache restores too).
    window.addEventListener("pageshow", function () {
      const changes = storageGet(FEEDBACK_KEY);
      if (!changes) return;
      Object.keys(changes).forEach(function (id) {
        const card = feed.querySelector('.card[data-id="' + id + '"]');
        if (card) applyState(card, changes[id]);
      });
      storageRemove(FEEDBACK_KEY);
    });
  }

  // ---- misc ---------------------------------------------------------------

  document.querySelectorAll("form[data-confirm]").forEach(function (form) {
    form.addEventListener("submit", function (event) {
      if (!window.confirm(form.dataset.confirm)) event.preventDefault();
    });
  });

  if ("serviceWorker" in navigator) {
    window.addEventListener("load", function () {
      navigator.serviceWorker.register("/sw.js").catch(function () {});
    });
  }
})();
