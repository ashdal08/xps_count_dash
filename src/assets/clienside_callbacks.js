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
    check_running_progress: (not_running, save_filename_switch) => {
      if (!not_running) {
        return [
          true, false, true, "mb-3 card-disable", true, true, true, true, true
        ];
      } else {
        return [
          false, true, false, "mb-3", false, false, false, !save_filename_switch, false
        ];
      }
    },
    wind_up_after_measurement: (interval_disabled) => {
      return [!interval_disabled, !interval_disabled];
    },
    update_display_values: (
      current_kinetic_energy,
      current_binding_energy,
      elapsed_time,
      remaining_time
    ) => {
      return [
        current_kinetic_energy.toFixed(2),
        current_binding_energy.toFixed(2),
        elapsed_time.toFixed(2),
        remaining_time.toFixed(2)
      ];
    },
    confirm_file_save_modal_open: (n1, n2, is_open) => {
      if (n1 || n2) {
        return !is_open;
      }
      return is_open;
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
          true, // save-as-file valid
          false, // save-as-file invalid
          true, // save-on-complete-filename valid
          false, // save-on-complete-filename invalid
          !isSave ? window.dash_clientside.no_update : false // start-click disabled
        ];
      } else {
        return [
          isSave ? window.dash_clientside.no_update : true, // confirm-save disabled
          false,   // save-as-file valid
          true,  // save-as-file invalid
          false,  // save-on-complete-filename valid
          true, // save-on-complete-filename invalid
          isSave ? true : window.dash_clientside.no_update     // start-click disabled
        ];
      }
    },
  },
});
