window.dash_clientside = Object.assign({}, window.dash_clientside, {
  clientside: {
    theme_switched: (switchOn) => {
      document.documentElement.setAttribute(
        "data-bs-theme",
        switchOn ? "light" : "dark"
      );
      return window.dash_clientside.no_update;
    },
    save_on_complete_toggled: (switchOn) => {
      return [!switchOn, switchOn];
    },
    check_running_progress: (notRunning, saveFilenameSwitch) => {
      if (!notRunning) {
        return [
          true, false, true, "mb-3 card-disable", true, true, true, true, true
        ];
      } else {
        return [
          false, true, false, "mb-3", false, false, false, !saveFilenameSwitch, false
        ];
      }
    },
    wind_up_after_measurement: (intervalDisabled) => {
      return [!intervalDisabled, !intervalDisabled];
    },
    update_display_values: (
      currentKineticEnergy,
      currentBindingEnergy,
      elapsedTime,
      remainingTime,
      currentProgress
    ) => {
      return [
        currentProgress,
        currentKineticEnergy.toFixed(2),
        currentBindingEnergy.toFixed(2),
        elapsedTime.toFixed(2),
        remainingTime.toFixed(2),
      ];
    },
    confirm_file_save_modal_open: (n1, n2, isOpen) => {
      if (n1 || n2) {
        return !isOpen;
      }
      return isOpen;
    },
    check_filename_validity: (data) => {
      if (!data || !data.target) {
        return [window.dash_clientside.no_update, window.dash_clientside.no_update,
        window.dash_clientside.no_update, window.dash_clientside.no_update,
        window.dash_clientside.no_update, window.dash_clientside.no_update];
      }

      const isSave = data.target === "save-on-complete-filename";
      const isValid = data.valid;

      if (isValid) {
        return [
          isSave ? window.dash_clientside.no_update : false,  // confirm-save disabled
          isSave ? window.dash_clientside.no_update : true, // save-as-file valid
          isSave ? window.dash_clientside.no_update : false, // save-as-file invalid
          !isSave ? window.dash_clientside.no_update : true, // save-on-complete-filename valid
          !isSave ? window.dash_clientside.no_update : false, // save-on-complete-filename invalid
          !isSave ? window.dash_clientside.no_update : false // start-click disabled
        ];
      } else {
        return [
          isSave ? window.dash_clientside.no_update : true, // confirm-save disabled
          isSave ? window.dash_clientside.no_update : false,   // save-as-file valid
          isSave ? window.dash_clientside.no_update : true,  // save-as-file invalid
          !isSave ? window.dash_clientside.no_update : false,  // save-on-complete-filename valid
          !isSave ? window.dash_clientside.no_update : true, // save-on-complete-filename invalid
          !isSave ? window.dash_clientside.no_update : true    // start-click disabled
        ];
      }
    },
  },
});
