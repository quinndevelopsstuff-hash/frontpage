// Front Page client: feedback buttons, "opened" tracking, load more, PWA.
(function () {
  "use strict";

  const feed = document.getElementById("feed");
  const status = document.getElementById("status");

  function announce(text) {
    if (status) status.textContent = text;
  }

  function sendFeedback(articleId, kind, value, keepalive) {
    return fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ article_id: articleId, kind: kind, value: value }),
      keepalive: !!keepalive,
    }).then(function (response) {
      if (!response.ok) throw new Error("HTTP " + response.status);
      return response.json();
    });
  }

  function applyState(card, state) {
    const pressed = { up: state.up, down: state.down, save: state.saved };
    card.querySelectorAll("button[data-kind]").forEach(function (button) {
      button.setAttribute("aria-pressed", pressed[button.dataset.kind] ? "true" : "false");
    });
    card.classList.toggle("is-dismissed", !!state.down);
  }

  function currentState(card) {
    const state = {};
    card.querySelectorAll("button[data-kind]").forEach(function (button) {
      state[button.dataset.kind === "save" ? "saved" : button.dataset.kind] =
        button.getAttribute("aria-pressed") === "true";
    });
    return state;
  }

  const MESSAGES = {
    up: ["Marked: more like this", "Removed like"],
    down: ["Marked: less like this", "Removed dislike"],
    save: ["Saved for later", "Removed from saved"],
  };

  function onFeedbackClick(button) {
    const card = button.closest(".card");
    const kind = button.dataset.kind;
    const value = button.getAttribute("aria-pressed") !== "true";
    const before = currentState(card);

    // Optimistic update so the tap feels instant; rolled back on failure.
    const next = Object.assign({}, before);
    next[kind === "save" ? "saved" : kind] = value;
    if (value && kind === "up") next.down = false;
    if (value && kind === "down") next.up = false;
    applyState(card, next);
    button.disabled = true;

    sendFeedback(Number(card.dataset.id), kind, value)
      .then(function (state) {
        applyState(card, state);
        announce(MESSAGES[kind][value ? 0 : 1]);
      })
      .catch(function () {
        applyState(card, before);
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
    const link = event.target.closest(".card-link");
    if (link) recordOpen(link);
  });
  // Middle-click also opens the article.
  document.addEventListener("auxclick", function (event) {
    const link = event.target.closest(".card-link");
    if (link && event.button === 1) recordOpen(link);
  });

  function recordOpen(link) {
    const card = link.closest(".card");
    if (!card || card.dataset.opened) return;
    card.dataset.opened = "1";
    sendFeedback(Number(card.dataset.id), "open", true, true).catch(function () {
      delete card.dataset.opened;
    });
  }

  // If a cached image is missing or broken, drop it: the typographic cover
  // underneath shows instead.
  document.addEventListener(
    "error",
    function (event) {
      const target = event.target;
      if (target.tagName === "IMG" && target.closest(".card-media")) target.remove();
    },
    true
  );

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
