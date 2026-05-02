(() => {
  const logoutForms = document.querySelectorAll("[data-auth-logout-form]");

  logoutForms.forEach((form) => {
    form.addEventListener("submit", async (event) => {
      event.preventDefault();

      const redirectPath = form.dataset.redirectPath || "/login";
      try {
        await fetch(form.action, {
          method: "POST",
          credentials: "include",
        });
      } catch (_error) {
        // Network failures should not trap the user on the logout screen.
      } finally {
        window.location.assign(redirectPath);
      }
    });
  });
})();
