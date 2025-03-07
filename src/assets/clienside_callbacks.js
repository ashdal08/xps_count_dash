window.dash_clientside = Object.assign({}, window.dash_clientside, {
  clientside: {
    theme_switched: (switchOn) => {
      document.documentElement.setAttribute(
        "data-bs-theme",
        switchOn ? "light" : "dark"
      );
      return window.dash_clientside.no_update;
    },
  },
});
