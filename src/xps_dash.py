import math
import re
import time
import os
import numpy as np
import pandas as pd
import polars as pl

import threading


import plotly.graph_objects as go
import plotly.io as pio

from gevent.pywsgi import WSGIServer
import waitress

# import u6
from dependencies import dummy_labjack_u6 as u6

from dash import (
    Dash,
    html,
    Input,
    Output,
    callback,
    dcc,
    clientside_callback,
    Patch,
    State,
    ClientsideFunction,
    dash_table,
)
import dash_bootstrap_components as dbc

from dash_bootstrap_templates import load_figure_template

load_figure_template(["bootstrap", "bootstrap_dark"])

BOOTSTRAP = pio.templates["bootstrap"]
BOOTSTRAP_DARK = pio.templates["bootstrap_dark"]

LAYOUT = {
    "autosize": True,
    "margin": {
        "l": 80,
        "r": 5,
        "b": 40,
        "t": 40,
    },
    "dragmode": "pan",
    "font_size": 11,
    "hovermode": "closest",
    "template": BOOTSTRAP,
    "modebar": {"orientation": "v"},
}
"""Dictionary describing the layout for the plot area of the dash app."""

CONFIG: dict = {
    "scrollZoom": True,
    "displaylogo": False,
    "toImageButtonOptions": {
        "format": "svg",
        "filename": "custom_image",
        "scale": 1,
    },
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
}
"""Dictionary describing the configuration of the plotly Figure object."""

THEMES: list = [
    "bootstrap",
    "cerulean",
    "cosmo",
    "cyborg",
    "darkly",
    "flatly",
    "journal",
    "litera",
    "lumen",
    "lux",
    "materia",
    "minty",
    "morph",
    "pulse",
    "quartz",
    "sandstone",
    "simplex",
    "sketchy",
    "slate",
    "solar",
    "spacelab",
    "superhero",
    "united",
    "vapor",
    "yeti",
    "zephyr",
]
"""List of theme names for the dash app."""


def reset_fig(theme_dark: bool = False, batch_mode: bool = False) -> go.Figure:
    """Method to create blank plot figure.

    Parameters
    ----------
    theme_dark : bool, optional
        Boolean indicating if the figure has to use a dark theme template, by default False
    batch_mode : bool, optional
        Boolean indicating if the measurement type is a batch mode measurement, by default False

    Returns
    -------
    plotly.graph_objects.Figure
        The plotly figure object.
    """
    trace_name = "pass_1"
    if batch_mode:
        trace_name = "b1_pass_1"
    fig = go.Figure(
        data=[go.Scatter(x=[], y=[], mode="markers+lines", line_width=1, marker_size=4, name=trace_name)],
        layout=go.Layout(
            **LAYOUT,
            xaxis_title="Binding Energy [eV]",
            yaxis_title="Counts",
            uirevision="no_change",
            legend={
                "orientation": "h",
                "xanchor": "center",
                "x": 0.5,
            },
        ),
    )

    fig = addPlotRefLines(fig)

    if theme_dark:
        fig.update_layout(dict(template=BOOTSTRAP_DARK))
        fig = updateRefLinesTheme(fig, True)

    return fig


def addPlotRefLines(fig: go.Figure) -> go.Figure:
    """Method to add reference photoelectron spectral lines to the plot figure.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The plotly graph object in which plot reference lines are to be added.

    Returns
    -------
    plotly.graph_objects.Figure
        The plotly Figure object after adding the reference lines.
    """
    fig.add_vline(
        x=83.95,
        line=dict(
            color="#B3DF72",
            width=3,
            dash="dash",
        ),
        opacity=0.6,
        name=r"$\large \text{Au 4f}_\text{7/2}$",
        showlegend=True,
    )
    fig.add_vline(
        x=87.9,
        line=dict(
            color="#FD9891",
            width=3,
            dash="dash",
        ),
        opacity=0.6,
        name=r"$\large \text{Au 4f}_\text{5/2}$",
        showlegend=True,
    )
    fig.add_vline(
        x=284.4,
        line=dict(
            color="#FEC195",
            width=3,
            dash="dash",
        ),
        opacity=0.6,
        name=r"$\large \text{C 1s}$",
        showlegend=True,
    )
    fig.add_vline(
        x=72.84,
        line=dict(
            color="#F5CD47",
            width=3,
            dash="dash",
        ),
        opacity=0.6,
        name=r"$\large \text{Al 2p}_\text{3/2}$",
        showlegend=True,
    )
    fig.add_vline(
        x=532.70,
        line=dict(
            color="#7EE2B8",
            width=3,
            dash="dash",
        ),
        opacity=0.6,
        name=r"$\large \text{O 1s}$",
        showlegend=True,
    )
    fig.add_vline(
        x=118,
        line=dict(
            color="#9DD9EE",
            width=3,
            dash="dash",
        ),
        opacity=0.6,
        name=r"$\large \text{Al 2s}$",
        showlegend=True,
    )

    return fig


def updateRefLinesTheme(fig: go.Figure, dark_theme: bool = False):
    """Method to change the colors of the reference lines when the theme template is changed in the plotly Fgiure between dark and light.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The plotly Figure object in which the reference lines exist.
    dark_theme : bool, optional
        Boolean indicating if the dark theme template of the Figure is on., by default False

    Returns
    -------
    plotly.graph_objects.Figure
        The plotly Figure object with the updated colors of the reference lines.
    """
    if dark_theme:
        fig.update_traces(
            overwrite=False,
            line=dict(
                color="#B3DF72",
            ),
            selector=dict(name=r"$\large \text{Au 4f}_\text{7/2}$"),
        )
        fig.update_traces(
            overwrite=False,
            line=dict(
                color="#FD9891",
            ),
            selector=dict(name=r"$\large \text{Au 4f}_\text{5/2}$"),
        )
        fig.update_traces(
            overwrite=False,
            line=dict(
                color="#FEC195",
            ),
            selector=dict(name=r"$\large \text{C 1s}$"),
        )
        fig.update_traces(
            overwrite=False,
            line=dict(
                color="#F5CD47",
            ),
            selector=dict(name=r"$\large \text{Al 2p}_\text{3/2}$"),
        )
        fig.update_traces(
            overwrite=False,
            line=dict(
                color="#7EE2B8",
            ),
            selector=dict(name=r"$\large \text{O 1s}$"),
        )
        fig.update_traces(
            overwrite=False,
            line=dict(
                color="#9DD9EE",
            ),
            selector=dict(name=r"$\large \text{Al 2s}$"),
        )
    else:
        fig.update_traces(
            overwrite=False,
            line=dict(
                color="#5B7F24",
            ),
            selector=dict(name=r"$\large \text{Au 4f}_\text{7/2}$"),
        )
        fig.update_traces(
            overwrite=False,
            line=dict(
                color="#C9372C",
            ),
            selector=dict(name=r"$\large \text{Au 4f}_\text{5/2}$"),
        )
        fig.update_traces(
            overwrite=False,
            line=dict(
                color="#C25100",
            ),
            selector=dict(name=r"$\large \text{C 1s}$"),
        )
        fig.update_traces(
            overwrite=False,
            line=dict(
                color="#946F00",
            ),
            selector=dict(name=r"$\large \text{Al 2p}_\text{3/2}$"),
        )
        fig.update_traces(
            overwrite=False,
            line=dict(
                color="#1F845A",
            ),
            selector=dict(name=r"$\large \text{O 1s}$"),
        )
        fig.update_traces(
            overwrite=False,
            line=dict(
                color="#227D9B",
            ),
            selector=dict(name=r"$\large \text{Al 2s}$"),
        )

    return fig


def addOrUpdatePlotTraceData(
    fig: go.Figure, pass_index: int, x_data: pd.Series, y_data: pd.Series, trace_name: str
) -> go.Figure:
    """Method to add or update traces in the plotly Figure object. To be used whenever a new point is obtained from the instrument.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The plotly Figure object in which the traces are to be added or updated.
    pass_index : int
        Integer indicating the index of the trace to be added or updated.
    x_data : pandas.Series
        The data for the x-coordinate of the trace points.
    y_data : pandas.Series
        The data for the y-coordinate of the trace points.
    trace_name : str
        The name of the trace to be added or updated.

    Returns
    -------
    plotly.graph_objects.Figure
        The plotly Figure object returned after adding or updating the trace.
    """
    existing_passes = len(fig["data"])
    if existing_passes < pass_index:
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_data,
                mode="markers+lines",
                line_width=1,
                marker_size=4,
                name=trace_name,
            )
        )
    fig.update_traces(
        overwrite=False,
        x=x_data,
        y=y_data,
        selector=dict(name=trace_name),
    )

    return fig


class DataBackend:
    """Class for the GUI backend.

    Methods
    -------
    connectLabJack:
        Connect to the Labjack U6 device.
    bindingEnergyToVolt:
        Convert the binding energy to voltage for the MAX5216 DAC.
    setSpiVoltage:
        Set the voltage at the MAX5216 DAC.
    startMeasurement:
        Start the measurement process.
    runSingleMeasurement:
        Run a single measurement.
    runBatchMeasurement:
        Run a batch measurement.
    saveMeasurementData:
        Save the measurement data to a CSV file.
    interruptionClicked:
        Handle the interruption of the measurement.
    onClose:
        Method fired when the close button is clicked.
    """

    BATCH_START_EV = 0
    """Index of the start energy value in the batch input table."""
    BATCH_END_EV = 1
    """Index of the end energy value in the batch input table."""
    BATCH_STEP_EV = 2
    """Index of the step energy value in the batch input table."""
    BATCH_TIME_PER_STEP = 3
    """Index of the time per step value in the batch input table."""
    BATCH_PASSES = 4
    """Index of the number of passes value in the batch input table."""
    EXCITATION_AL = 1486.6
    """The excitation energy for Aluminium cathode."""
    EXCITATION_MG = 1253.6
    """The excitation energy for Magnesium cathode."""
    U6_MOSI_PIN_NUM = 2
    """The MOSI pin number for the Labjack U6."""
    U6_MISO_PIN_NUM = 4
    """The MISO pin number for the Labjack U6."""
    U6_CLK_PIN_NUM = 1
    """The CLK pin number for the Labjack U6."""
    U6_CS_PIN_NUM = 3
    """The CS pin number for the Labjack U6."""
    DEP_PATH = ".\\src\\dependencies\\"
    """Path for all the files that the program requires to function."""

    u6_labjack: u6.U6
    """Lajack U6 object for the measurement."""
    labjack_connect: bool
    """Boolean to check if the Labjack is connected."""
    data_table: pl.DataFrame
    """Polars DataFrame to store the measurement data."""
    pass_row_indexes: list
    """List of row indices indicating start of new passes."""
    typ_schema: dict
    """Dictionary indicating the scheme for the types in the polars dataframe."""
    measurement_thread: threading.Thread
    """Thread object to run the measurement."""
    excitation_voltage: float
    """Selected excitation voltage for the measurement."""
    plot_fig: go.Figure
    """Plotly figure object to plot the measurement data."""
    plot_file_name: str
    """HTML plot file name."""
    meas_interrupted: bool = False
    """Boolean to check if the measurement was interrupted."""
    meas_interrupt_id: int
    """ID of the button that was clicked to interrupt the measurement."""
    meas_running: bool = False
    """Boolean to check if the measurement is running."""
    current_binding_energy: float = 0
    """Variable holding the value of the current binding energy of the measurement."""
    current_kinetic_energy: float = 0
    """Variable holding the current kinetic energy of the measurement."""
    setpoint_ev: float
    """Duplicate variable for the current binding energy of the measurement."""
    current_progress: int = 0
    """Integer indicating the current progress percentage of the measurement."""
    remaining_time: float = 0
    """Variable with the remaining time for the measurement in minutes."""
    elapsed_time: float = 0
    """Variable with the elapsed time of the measurement in minutes."""
    total_steps: int = 0
    """Integer indicating the total number of steps in the measurement."""
    total_batch_passes: int = 0
    """Integer holding the total number of passes during the batch mode measurement."""
    batch_pass_no: int = 1
    """Integer holding the current pass number during a batch measurement."""
    batch_step_no: int = 1
    """Integer holding the current step number during a batch measurement."""
    meas_completed: bool = False
    """Boolean indicating if a measurement is completed."""

    def __init__(self) -> None:
        """
        Initialize the main window of the program.

        """
        self.connectLabjack()
        self.plot_fig = reset_fig()

        self.plot_file_name = "".join([os.getcwd(), "\\src\\dependencies\\", "plt_dat.html"])

    def connectLabjack(self) -> None:
        """Connect to the Labjack U6 device."""
        self.u6_labjack = u6.U6()
        self.u6_labjack.getCalibrationData()
        self.u6_labjack.configIO(EnableCounter0=True)
        # ljud.eDAC(self.u6_labjack, 1, 3)  # Set the reference voltage for the MAX5216 DAC
        self.u6_labjack.getFeedback(u6.DAC1_8(self.u6_labjack.voltageToDACBits(3, 1)))
        self.labjack_connect = True

    def bindingEnergyToVolt(self, binding_energy: float) -> float:
        """Convert the binding energy to voltage for the MAX5216 DAC.

        Parameters
        ----------
        binding_energy : float
            The binding energy value.

        Returns
        -------
        float
            The voltage value for the MAX5216 DAC.

        """
        self.setpoint_ev = binding_energy
        kinetic_energy = self.excitation_voltage - binding_energy
        self.current_binding_energy = binding_energy
        self.current_kinetic_energy = kinetic_energy
        return kinetic_energy / 498.43

    def setSpiVoltage(self, volt: float) -> None:
        """Set the voltage at the MAX5216 DAC.

        Parameters
        ----------
        volt : float
            The voltage value to set at the MAX5216 DAC.

        """
        ref_volt_at_MAX5216_DAC = self.u6_labjack.getAIN(3)  # the reference voltage at the DAC MAX5216
        volt_bits = int(
            volt * 65535 / ref_volt_at_MAX5216_DAC
        )  # convert input volts to a 16 bit scaled value with the reference voltage
        volt_bits = (
            volt_bits << 6
        )  # shift the 16 bit value 6 bits to the left to make it a 22 bit value. Refer to MAX5216 datasheet
        volt_bits |= (
            1 << 22
        )  # set the 23rd bit to 1 to indicate that the first 2 bits are the command bits. Refer to MAX5216 datasheet
        self.u6_labjack.spi(
            list(volt_bits.to_bytes(3)),
            CSPINNum=self.U6_CS_PIN_NUM,
            CLKPinNum=self.U6_CLK_PIN_NUM,
            MISOPinNum=self.U6_MISO_PIN_NUM,
            MOSIPinNum=self.U6_MOSI_PIN_NUM,
        )
        # Set the voltage at DAC0 of the Labjack for comparison with MAX5216.
        self.u6_labjack.getFeedback(u6.DAC0_8(self.u6_labjack.voltageToDACBits(volt, 0)))

    def startMeasurement(
        self,
        start_ev: float,
        end_ev: float,
        step_ev: float,
        time_per_step: float,
        pass_no: int,
        batch_dataframe: pd.DataFrame,
        source_mg: bool,
        batch_mode: bool = False,
    ) -> None:
        """Start the measurement process.

        Parameters
        ----------
        start_ev : float
            The start value for the binding energy in the single mode measurement in eV.
        end_ev : float
            The end value of the binding energy in the single mode measurement in eV.
        step_ev : float
            The step increment value to be used in the single mode measurement in eV.
        time_per_step : float
            The time to wait per step during the single mode measurement in s.
        pass_no : int
            The no. of passes to be measured during a single mode measurement.
        batch_dataframe : pandas.DataFrame
            The pandas Dataframe formed from the dat in the batch mode grid.
        source_mg : bool
            Boolean indicating if the source of the X-ray excitation is Mg.
        batch_mode : bool, optional
            Boolean indicating if the measurement is a batch mode measurement, by default False
        """
        self.total_steps = 0
        self.total_batch_passes = 0
        self.batch_pass_no = 1
        self.batch_step_no = 1
        if source_mg:
            self.excitation_voltage = self.EXCITATION_MG
        else:
            self.excitation_voltage = self.EXCITATION_AL

        if batch_mode:
            # if Batch Scan tab is selected in the UI. Batch scan mode is 1. Single scan mode is 0.

            total_time_s = 0

            for row_index in range(0, len(batch_dataframe)):
                batch_time = 0
                start_ev = batch_dataframe["Start [eV]"][row_index]
                end_ev = batch_dataframe["End [eV]"][row_index]
                step_ev = batch_dataframe["Step [eV]"][row_index]
                time_per_step = batch_dataframe["Time [s/eV]"][row_index]
                pass_no = batch_dataframe["Passes"][row_index]
                if math.isnan(start_ev):
                    break
                batch_time = (time_per_step) * (pass_no) * ((end_ev - start_ev) / step_ev + 1)
                total_time_s += batch_time
                self.total_steps += int((end_ev - start_ev) / step_ev + 1) * pass_no
                self.total_batch_passes += pass_no

            self.elapsed_time = 0
            self.remaining_time = round(total_time_s / 60, 2)
            self.current_progress = 0
            self.measurement_thread = threading.Thread(target=self.runBatchMeasurement, args=[batch_dataframe])
            self.plot_fig = reset_fig(batch_mode=True)
        else:
            self.total_steps = int(abs((end_ev - start_ev) / step_ev + 1) * pass_no)
            total_time_s = time_per_step * pass_no * ((end_ev - start_ev) / step_ev + 1)
            self.remaining_time = round(total_time_s / 60, 2)
            self.elapsed_time = 0
            self.current_progress = 0

            self.measurement_thread = threading.Thread(
                target=self.runSingleMeasurement,
                args=[start_ev, end_ev, step_ev, time_per_step, pass_no, 0, 0, False],
            )
            self.plot_fig = reset_fig()

        self.typ_schema = {
            "Binding Energy [eV]": pl.Float64,
            "Pass No.": pl.Float64,
            "Time_per_step [s]": pl.Float64,
            "Counts": pl.Float64,
            "Counts_per_milli [/ms]": pl.Float64,
            "U6_DAC0_Voltage [V]": pl.Float64,
            "MAX5216_DAC_Voltage [V]": pl.Float64,
        }
        data_temp = {
            "Binding Energy [eV]": [],
            "Pass No.": [],
            "Time_per_step [s]": [],
            "Counts": [],
            "Counts_per_milli [/ms]": [],
            "U6_DAC0_Voltage [V]": [],
            "MAX5216_DAC_Voltage [V]": [],
        }

        self.data_table = pl.DataFrame(
            data_temp,
            schema=self.typ_schema,
        )

        self.meas_running = True
        self.meas_completed = False
        self.meas_interrupted = False

        self.measurement_thread.start()

    def runSingleMeasurement(
        self,
        start_ev: float,
        end_ev: float,
        step_ev: float,
        time_per_step: float,
        pass_no: int,
        batch_no: int,
        batch_pass_no: int,
        type_batch=False,
    ) -> None:
        """Run a single measurement.

        Parameters
        ----------
        start_ev : float
            The starting energy value for the measurement in eV.
        end_ev : float
            The ending energy value for the measurement in eV.
        step_ev : float
            The step width for the measurement in eV.
        time_per_step : float
            The time per step for the measurement in s.
        pass_no : int
            The number of passes for the measurement.
        batch_no : int
            The current batch number of the measurement if in batch mode.
        batch_pass_no : int
            The current pass no. of the meaasurement if in batch mode.
        type_batch : bool
            The type of measurement. True for batch measurement, False for single measurement, by default False.

        """
        self.setSpiVoltage(self.bindingEnergyToVolt(start_ev))

        pass_index = 1

        plot_dataframe = pd.DataFrame({
            "Binding Energy [eV]": [],
            "Counts": [],
        })

        step_no = 1

        while pass_index <= pass_no:
            self.u6_labjack.getFeedback(u6.Counter0(True))
            start_time = time.time()
            __refresh_time = 0
            while __refresh_time <= time_per_step:
                if self.meas_interrupted:
                    break
                if __refresh_time + 1 < time_per_step:
                    time.sleep(1)
                    __refresh_time += 1
                else:
                    time.sleep(time_per_step - __refresh_time)
                    break

            if self.meas_interrupted:
                self.meas_interrupted = False
                self.meas_completed = True
                self.meas_running = False
                self.current_progress = 100
                time.sleep(1)
                return
            counts = self.u6_labjack.getFeedback(u6.Counter0(False))[0]
            time_taken = time.time() - start_time  # record in s
            binding_energy = self.setpoint_ev
            counts_per_milli = round(counts / (time_taken * 1000), 6)
            u6_dac0_voltage = round(self.u6_labjack.getAIN(0), 6)
            max5216_dac_voltage = round(self.u6_labjack.getAIN(2), 6)

            new_data = {
                "Binding Energy [eV]": binding_energy,
                "Pass No.": pass_index,
                "Time_per_step [s]": time_taken,
                "Counts": counts,
                "Counts_per_milli [/ms]": counts_per_milli,
                "U6_DAC0_Voltage [V]": u6_dac0_voltage,
                "MAX5216_DAC_Voltage [V]": max5216_dac_voltage,
            }

            new_plot_point = pd.DataFrame({
                "Binding Energy [eV]": [binding_energy],
                "Counts": [counts],
            })

            plot_dataframe = pd.concat([plot_dataframe, new_plot_point])

            new_df = pl.DataFrame(new_data, schema=self.typ_schema)
            self.data_table = pl.concat([self.data_table, new_df])
            if not type_batch:
                self.plot_fig = addOrUpdatePlotTraceData(
                    self.plot_fig,
                    pass_index,
                    plot_dataframe["Binding Energy [eV]"],
                    plot_dataframe["Counts"],
                    f"pass_{pass_index}",
                )
            else:
                self.plot_fig = addOrUpdatePlotTraceData(
                    self.plot_fig,
                    self.batch_pass_no,
                    plot_dataframe["Binding Energy [eV]"],
                    plot_dataframe["Counts"],
                    f"b{batch_no}_pass_{pass_index}",
                )
            # if not type_batch:
            #     if existing_passes < pass_index:
            #         self.plot_fig.add_trace(
            #             go.Scatter(
            #                 x=plot_dataframe["Binding Energy [eV]"],
            #                 y=plot_dataframe["Counts"],
            #                 mode="markers+lines",
            #                 line_width=1,
            #                 marker_size=4,
            #                 name=f"pass_{pass_index}",
            #             )
            #         )
            #     self.plot_fig.update_traces(
            #         overwrite=False,
            #         x=plot_dataframe["Binding Energy [eV]"],
            #         y=plot_dataframe["Counts"],
            #         selector=dict(name=f"pass_{pass_index}"),
            #     )
            # else:
            #     if existing_passes < self.batch_pass_no:
            #         self.plot_fig.add_trace(
            #             go.Scatter(
            #                 x=plot_dataframe["Binding Energy [eV]"],
            #                 y=plot_dataframe["Counts"],
            #                 mode="markers+lines",
            #                 line_width=1,
            #                 marker_size=4,
            #                 name=f"b{batch_no}_pass_{pass_index}",
            #             )
            #         )
            #     self.plot_fig.update_traces(
            #         overwrite=False,
            #         x=plot_dataframe["Binding Energy [eV]"],
            #         y=plot_dataframe["Counts"],
            #         selector=dict(name=f"b{batch_no}_pass_{pass_index}"),
            #     )
            self.plot_fig.write_html(
                self.plot_file_name,
                config=CONFIG,
                include_plotlyjs="cdn",
                include_mathjax="cdn",
            )

            if type_batch:
                step_no = self.batch_step_no
            self.current_progress = int(step_no * 100 / self.total_steps)
            self.remaining_time -= time_taken / 60
            self.elapsed_time += time_taken / 60

            if self.setpoint_ev + step_ev <= end_ev:
                self.setpoint_ev += step_ev
                step_no += 1
                if type_batch:
                    self.batch_step_no += 1
                self.setSpiVoltage(self.bindingEnergyToVolt(self.setpoint_ev))
            else:
                self.setSpiVoltage(self.bindingEnergyToVolt(start_ev))
                pass_index += 1
                if type_batch:
                    self.batch_pass_no += 1
                    self.batch_step_no += 1
                plot_dataframe = pd.DataFrame({
                    "Binding Energy [eV]": [],
                    "Counts": [],
                })

        if not type_batch:
            self.current_progress = 100
            self.remaining_time = 0.0
            time.sleep(1)
            self.meas_completed = True
            self.meas_running = False

    def runBatchMeasurement(self, batch_dataframe: pd.DataFrame) -> None:
        """Run a batch measurement.

        Parameters
        ----------
        batch_dataframe : pandas.DataFrame
            The pandas Dataframe with the measurement parameters from the batch mode grid.
        """
        for row_index in range(0, len(batch_dataframe)):
            if self.meas_running:
                start_ev = batch_dataframe["Start [eV]"][row_index]
                end_ev = batch_dataframe["End [eV]"][row_index]
                step_ev = batch_dataframe["Step [eV]"][row_index]
                time_per_step = batch_dataframe["Time [s/eV]"][row_index]
                pass_no = batch_dataframe["Passes"][row_index]
                if math.isnan(start_ev):
                    break
                self.runSingleMeasurement(
                    start_ev, end_ev, step_ev, time_per_step, pass_no, row_index + 1, self.batch_pass_no, True
                )
        self.current_progress = 100
        self.remaining_time = 0.0
        time.sleep(1)
        self.meas_running = False
        self.meas_completed = True
        self.batch_pass_no = 1
        self.batch_step_no = 0
        return

    def saveMeasurementData(self, filename: str = "xps_data") -> None:
        """Save the measurement data to a CSV file.

        Parameters
        ----------
        filename : str, optional
            File name to save the results and plot as, by default "xps_data"
        """

        self.data_table.write_csv(filename + ".csv")

        self.current_progress = 100
        return

    def interruptionClicked(self) -> None:
        """Handle the interruption of the measurement."""
        self.meas_interrupted = True
        self.meas_running = False

    def onClose(self) -> None:
        """Method fired when the Shutdown App button is clicked."""
        os._exit(0)


data_backend = DataBackend()
"""The data processing backend class."""

fig = reset_fig()
"""The plotly figure to be displayed in the plot area of the dash app."""


app = Dash(
    __name__,  # external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME], #now served locally in the assets folder
    title="XPS",
)
"""The dash app variable."""

color_mode_switch = html.Span(
    [
        dbc.Label(
            className="fa fa-moon",
            html_for="switch",
        ),
        dbc.Switch(
            id="switch",
            value=True,
            className="d-inline-block ms-1",
            persistence=True,
        ),
        dbc.Label(
            className="fa fa-sun",
            html_for="switch",
        ),
    ],
)
"""A dash html Span element for a theme switching button."""

offcanvas = html.Div([
    dbc.Offcanvas(
        dcc.Markdown(
            "",
            mathjax=True,
            className="offcanvas-markdown",
        ),
        id="offcanvas",
        title="Impedance Spectroscopy",
        is_open=False,
        class_name="offcanvas-style",
    ),
])
"""A dash html div for a canvas that can be rolled into sight when called. Here theory of the impedance analysis is mentioned."""

save_file_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Save Files as")),
        dbc.ModalBody(
            dbc.Input(
                id="save-as-file",
                placeholder="Filename without extensions...",
                type="text",
                value="xps_data",
                pattern=r'[^.\\/:*?"<>|]+',
                # title=r'Filename should not contain .\\/:*?"\'<>|',
            ),
        ),
        dbc.ModalFooter([
            dbc.Button(
                "Save",
                id="confirm-save",
                className="ms-auto",
                n_clicks=0,
                outline=True,
                color="success",
            ),
            dbc.Button(
                "Cancel",
                id="cancel-save",
                className="ms-auto",
                n_clicks=0,
                outline=True,
                color="danger",
            ),
        ]),
    ],
    id="modal",
    is_open=False,
)
"""Dash modal to appear whenever the save button is clicked."""


def generate_single_mode_tab_content(
    start_ev: float, end_ev: float, ev_step: float, time_step: float, pass_no: int
) -> list:
    """Method to generate the content for the Dash container when the single mode tab is clicked.

    Parameters
    ----------
    start_ev : float
        The start value in eV to be shown in the start eV field of the Dash app.
    end_ev : float
        The end value in eV to be shown in the end eV field of the Dash app.
    ev_step : float
        The step value in eV to be shown in the step eV field of the Dash app.
    time_step : float
        The time per step in s to be shown in the time per step field of the Dash app.
    pass_no : int
        The pass no. to be shown in the no. of passes field in the Dash app.

    Returns
    -------
    list
        A list indicating the child components to be added to the respective Dash container.
    """
    return [
        dbc.Col([
            dbc.InputGroup(
                [
                    dbc.InputGroupText(
                        "Start : ",
                    ),
                    dbc.Input(
                        type="number",
                        step=0.001,
                        id="start-ev",
                        value=start_ev,
                        min=0.0,
                        max=1486.6,
                    ),
                    dbc.InputGroupText(
                        "eV",
                    ),
                ],
            ),
            dbc.InputGroup(
                [
                    dbc.InputGroupText(
                        "End : ",
                    ),
                    dbc.Input(
                        type="number",
                        step=0.001,
                        id="end-ev",
                        value=end_ev,
                        min=0.001,
                        max=1486.6,
                    ),
                    dbc.InputGroupText(
                        "eV",
                    ),
                ],
            ),
            dbc.InputGroup(
                [
                    dbc.InputGroupText(
                        "Step width : ",
                    ),
                    dbc.Input(
                        type="number",
                        step=0.001,
                        id="ev-step",
                        value=ev_step,
                        min=0.001,
                        max=1486.6,
                    ),
                    dbc.InputGroupText(
                        "eV",
                    ),
                ],
            ),
            dbc.InputGroup(
                [
                    dbc.InputGroupText(
                        "Time per step : ",
                    ),
                    dbc.Input(
                        type="number",
                        step=1.0,
                        id="time-step",
                        value=time_step,
                        min=1.0,
                    ),
                ],
            ),
            dbc.InputGroup(
                [
                    dbc.InputGroupText(
                        "No. of passes : ",
                    ),
                    dbc.Input(
                        type="number",
                        step=1,
                        id="meas-passes",
                        value=pass_no,
                        min=1,
                    ),
                ],
            ),
        ])
    ]


single_mode_tab_content = generate_single_mode_tab_content(0.0, 1000.0, 0.6, 1.0, 1)
"""The Dash content to be shown when the Single mode tab is clicked."""


blank_data_dict = {
    " ": [i for i in range(1, 51)],
    "Start [eV]": [np.nan for i in range(1, 51)],
    "End [eV]": [np.nan for i in range(1, 51)],
    "Step [eV]": [np.nan for i in range(1, 51)],
    "Time [s/eV]": [np.nan for i in range(1, 51)],
    "Passes": [np.nan for i in range(1, 51)],
}
"""A dictionary to generate black data for the batch mode grid in the Dash app."""

blank_data = pd.DataFrame(blank_data_dict)
"""The pandas Dataframe formed from the black_data_dict."""


def generate_batch_data_grid(dataframe: pd.DataFrame) -> list:
    """Method to generate the content for the Dash container when the batch mode tab is clicked.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The pandas Dataframe with the data for the batch mode grid in the Dash app.

    Returns
    -------
    list
        A list with the child entries for the Dash container.
    """
    return [
        dbc.Container(
            [
                dash_table.DataTable(
                    id="batch-seq",
                    columns=[
                        {
                            "name": i,
                            "id": i,
                            "editable": False if i in " " else True,
                            "type": "numeric",
                        }
                        for i in dataframe.columns
                    ],
                    data=dataframe.to_dict("records"),
                    editable=True,
                    fixed_rows={"headers": True},
                    # filter_action="native",
                    # sort_action="native",
                    style_table={
                        "overflowX": "auto",
                        "overflowY": "auto",
                        "border": "1px solid var(--bs-border-color-translucent)",
                        "height": "266px",
                        "scrollbar-color": "var(--bs-border-color) transparent",
                        "scrollbar-width": "thin",
                    },
                    style_header={
                        "border": "1px solid var(--bs-border-color-translucent)",
                        "border-top": "0px",
                        "background-color": "var(--bs-card-cap-bg)",
                        "text-color": "var(--bs-body-color)",
                        "whiteSpace": "normal",
                        "height": "auto",
                        "lineHeight": "18px",
                    },
                    style_data={
                        "border": "1px solid var(--bs-border-color-translucent)",
                        "background-color": "var(--bs-body-bg)",
                        "text-color": "var(--bs-body-color)",
                    },
                    style_data_conditional=[
                        {
                            "if": {
                                "state": "active",
                            },
                            "backgroundColor": "var(--bs-focus-ring-color)",
                            "border": "1px solid var(--bs-primary)",
                            "text-color": "var(--bs-body-color)",
                        },
                        {
                            "if": {
                                "column_id": " ",
                            },
                            "border-left": "0px",
                        },
                    ],
                    style_cell={
                        "textAlign": "center",
                        "minWidth": "72px",
                        "width": "72px",
                        "maxWidth": "72px",
                        # "whiteSpace": "normal",
                    },
                    style_cell_conditional=[
                        {
                            "if": {
                                "column_id": " ",
                            },
                            "textAlign": "center",
                            "minWidth": "30px",
                            "width": "30px",
                            "maxWidth": "30px",
                            # "whiteSpace": "normal",
                        }
                    ],
                ),
            ],
            class_name="table-container-mod",
        )
    ]


batch_mode_tab_content = generate_batch_data_grid(blank_data)
"""The batch mode content to be shown in the Dash app when the batch mode tab is clicked."""

card = dbc.Card(
    [
        dbc.CardHeader(
            dbc.Tabs(
                [
                    dbc.Tab(label="Single Mode", tab_id="tab-1"),
                    dbc.Tab(label="Batch Mode", tab_id="tab-2"),
                ],
                id="card-tabs",
                active_tab="tab-1",
            )
        ),
        dbc.CardBody(
            html.P(id="card-content", className="card-text", children=batch_mode_tab_content),
            class_name="card-body-mod",
        ),
    ],
    class_name="mb-3",
    id="meas-select-card",
)
"""A card container to render the tabs for the single mode and batch mode measurements."""

source_selection = dbc.Stack(
    [
        dbc.Label(
            "X-Ray Source",
            class_name="label-center-mod",
            html_for="source-select",
            align="center",
        ),
        dbc.Switch(
            id="source-select",
            class_name="fs-1 form-check-input-mod",
            label_class_name="form-check-label-mod-xps fs-6",
            # label="On",
        ),
    ],
    direction="horizontal",
    class_name="mx-auto mb-3 justify-content-center",
)
"""Dash components for the switch to select the source of the X-ray excitation."""


status_displays = [
    dbc.Row(
        [
            dbc.Col(
                dbc.Label(
                    "K.E [eV]",
                    class_name="label-center-disp-mod",
                ),
                class_name="justify-content-center",
            ),
            dbc.Col(
                dbc.Label(
                    "B.E [eV]",
                    class_name="label-center-disp-mod",
                ),
                class_name="justify-content-center",
            ),
            dbc.Col(
                dbc.Label(
                    "Elapsed [min]",
                    class_name="label-center-disp-mod",
                ),
                class_name="justify-content-center",
            ),
            dbc.Col(
                dbc.Label(
                    "Remaining [min]",
                    class_name="label-center-disp-mod",
                ),
                class_name="justify-content-center",
            ),
        ],
    ),
    dbc.Row([
        dbc.Col(
            dbc.Input(
                id="kin-energy-disp",
                readonly=True,
            )
        ),
        dbc.Col(
            dbc.Input(
                id="binding-energy-disp",
                readonly=True,
            )
        ),
        dbc.Col(
            dbc.Input(
                id="time-elapsed-disp",
                readonly=True,
            )
        ),
        dbc.Col(
            dbc.Input(
                id="time-remain-disp",
                readonly=True,
            )
        ),
    ]),
]
"""A list with the child entries to render the status displays of the current binding energy, current kinetic energy, elapsed time and remaining time of the measurement."""


# mathjax = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'
# # app.scripts.append_script({ 'external_url' : mathjax })
app.layout = html.Div(
    [
        dcc.Store(id="check-running", data=False),
        dcc.Store("start-ev-value", data=0.0),
        dcc.Store("end-ev-value", data=1000.0),
        dcc.Store("ev-step-value", data=0.6),
        dcc.Store("time-step-value", data=1.0),
        dcc.Store("meas-passes-value", data=1),
        dcc.Store(id="start-mode", data="single"),
        dcc.Store("batch-seq-dataframe", data=blank_data.to_dict("records")),
        dcc.Interval(id="progress-interval", interval=300, disabled=True),
        dcc.Interval(id="graph-interval-component", interval=300, disabled=False),
        dcc.Interval(id="interval-component", interval=300, disabled=False),
        save_file_modal,
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1(
                            html.A(
                                "XPS Count",
                                id="open-offcanvas",
                            )
                        ),
                        offcanvas,
                    ],
                    width="auto",
                ),
            ],
            class_name="content-rows",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        color_mode_switch,
                        card,
                        source_selection,
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Button(
                                        "Start",
                                        color="primary",
                                        outline=True,
                                        id="start-click",
                                    ),
                                    className="d-grid gap-2 col-6 mx-auto",
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Stop!",
                                        color="warning",
                                        outline=True,
                                        id="stop-click",
                                        disabled=True,
                                    ),
                                    className="d-grid gap-2 col-6 mx-auto",
                                ),
                            ],
                            class_name="button-rows",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Button(
                                        "Save Plot / Results",
                                        color="success",
                                        outline=True,
                                        id="save-results",
                                        disabled=True,
                                    ),
                                    className="d-grid gap-2 col-6 mx-auto",
                                ),
                            ],
                            class_name="button-rows",
                        ),
                        dbc.Progress(
                            id="progress-bar",
                            value="0",
                            striped=True,
                            animated=True,
                            class_name="mb-3",
                        ),
                        dbc.Row(status_displays),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Button(
                                        "Shutdown App!",
                                        color="danger",
                                        outline=True,
                                        id="shutdown-app",
                                        disabled=False,
                                    ),
                                    className="d-grid gap-2 col-6 mx-auto",
                                ),
                            ],
                            class_name="shutdown-button-rows",
                            align="end",
                        ),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        dcc.Graph(
                            id="graph",
                            figure=fig,
                            config=CONFIG,
                            mathjax=True,
                            className="dcc-graph",
                        ),
                        dcc.Download(id="download-data"),
                        dcc.Download(id="download-plot"),
                    ],
                    width=8,
                ),
            ],
            class_name="content-rows",
        ),
    ],
    className="app-ui-division",
)

http_server = WSGIServer(
    ("", 8060),
    app.server,
)
"""The WSGI Production server to run the dash app."""

# ********************************************************************************
# ********************************************************************************
# ********************************************************************************
# ********************************************************************************
# ********************************************************************************


clientside_callback(
    ClientsideFunction("clientside", "theme_switched"),
    Output("switch", "id"),
    Input("switch", "value"),
    # prevent_initial_call=True,
)
"""Method called on the clientside when the theme switch is toggled."""

# ********************************************************************************


@app.callback(
    Output("card-content", "children"),
    Output("start-ev-value", "data"),
    Output("end-ev-value", "data"),
    Output("ev-step-value", "data"),
    Output("time-step-value", "data"),
    Output("meas-passes-value", "data"),
    Output("batch-seq-dataframe", "data"),
    Input("card-tabs", "active_tab"),
    State("card-content", "children"),
    State("start-ev-value", "data"),
    State("end-ev-value", "data"),
    State("ev-step-value", "data"),
    State("time-step-value", "data"),
    State("meas-passes-value", "data"),
    State("batch-seq-dataframe", "data"),
)
def tab_content(
    active_tab: str,
    old_mode: list,
    start_ev: float,
    end_ev: float,
    ev_step: float,
    time_step: float,
    pass_no: int,
    batch_dataframe: dict,
) -> tuple[list, float, float, float, float, int, dict | list]:
    """Method to update the Dash app content when the tabs are switched between single and batch mode.

    Parameters
    ----------
    active_tab : str
        String with the id of the active tab or the tab that was clicked.
    old_mode : list
        List of children of the Dash container that was rendered before clicking the tab change.
    start_ev : float
        The start eV value from the Store to be shown in start ev field of the Dash app if the single mode tab was clicked.
    end_ev : float
        The end eV value from the Store to be shown in end ev field of the Dash app if the single mode tab was clicked.
    ev_step : float
        The step eV value from the Store to be shown in step ev field of the Dash app if the single mode tab was clicked.
    time_step : float
        The the time per step value from the Store to be shown in time per step field of the Dash app if the single mode tab was clicked.
    pass_no : int
        The no. of passes value from the Store to be shown in no. of passes field of the Dash app if the single mode tab was clicked.
    batch_dataframe : dict
        The dictionary of values from the Store to be shown in the batch data grid if the batch mode tab was clicked.

    Returns
    -------
    tuple[list, float, float, float, float, int, dict]
        Returns the list of child entries to be render for the respective active tab, start ev value, end ev value, time per step value, no. of passes value, dictionary with the batch data grid, respectively.
    """
    if active_tab in "tab-2":
        card_content = generate_batch_data_grid(pd.DataFrame(batch_dataframe))
        start_ev_mod = old_mode[0]["props"]["children"][0]["props"]["children"][1]["props"]["value"]
        end_ev_mod = old_mode[0]["props"]["children"][1]["props"]["children"][1]["props"]["value"]
        ev_step_mod = old_mode[0]["props"]["children"][2]["props"]["children"][1]["props"]["value"]
        time_step_mod = old_mode[0]["props"]["children"][3]["props"]["children"][1]["props"]["value"]
        pass_no_mod = old_mode[0]["props"]["children"][4]["props"]["children"][1]["props"]["value"]
        return (
            card_content,
            start_ev_mod,
            end_ev_mod,
            ev_step_mod,
            time_step_mod,
            pass_no_mod,
            batch_dataframe,
        )
    else:
        batch_dataframe_mod = old_mode[0]["props"]["children"][0]["props"]["data"]
        card_content = generate_single_mode_tab_content(start_ev, end_ev, ev_step, time_step, pass_no)
        temp_batch_data = pd.DataFrame(batch_dataframe_mod).to_dict("records")
        return (
            card_content,
            start_ev,
            end_ev,
            ev_step,
            time_step,
            pass_no,
            temp_batch_data,
        )


# ********************************************************************************


@callback(
    Output("graph", "figure"),
    Input("switch", "value"),
    State("graph", "figure"),
    # prevent_initial_call=True,
)
def updateFigureTemplate(switch_on: int, ex_fig: dict) -> go.Figure | Patch:
    """Method called to change the theme of the figure when the theme of the app is changed.

    Parameters
    ----------
    switch_on : int
        Integer indicating the state of the switch. 0 is OFF and 1 is ON.

    Returns
    -------
    plotly.graph_objects.Figure | dash.Patch
        The plotly figure or dash Patch object to be sent to the graph division of the app.

    """

    # When using Patch() to update the figure template, you must use the figure template dict

    template = BOOTSTRAP if switch_on else BOOTSTRAP_DARK

    dark_theme = False if switch_on else True

    old_fig = go.Figure(ex_fig)

    old_fig = updateRefLinesTheme(old_fig, dark_theme)

    patched_figure = Patch()
    patched_figure["layout"]["template"] = template
    patched_figure["data"] = old_fig["data"]

    return patched_figure


# *********************************************************************************


@callback(
    Output(
        "start-mode",
        "data",
        allow_duplicate=True,
    ),
    Output(
        "start-ev-value",
        "data",
        allow_duplicate=True,
    ),
    Output(
        "end-ev-value",
        "data",
        allow_duplicate=True,
    ),
    Output(
        "ev-step-value",
        "data",
        allow_duplicate=True,
    ),
    Output(
        "time-step-value",
        "data",
        allow_duplicate=True,
    ),
    Output(
        "meas-passes-value",
        "data",
        allow_duplicate=True,
    ),
    Output(
        "batch-seq-dataframe",
        "data",
        allow_duplicate=True,
    ),
    Input("start-click", "n_clicks"),
    State("card-content", "children"),
    State("card-tabs", "active_tab"),
    State("start-ev-value", "data"),
    State("end-ev-value", "data"),
    State("ev-step-value", "data"),
    State("time-step-value", "data"),
    State("meas-passes-value", "data"),
    State("batch-seq-dataframe", "data"),
    prevent_initial_call=True,
)
def prepareForMeasurement(
    n: int,
    card_body: list,
    mode: str,
    start_ev: float,
    end_ev: float,
    ev_step: float,
    time_step: float,
    pass_no: int,
    batch_dataframe: dict,
) -> tuple[str, float, float, float, float, int, dict]:
    """Method called when the Start button is clicked.

    Parameters
    ----------
    n : int
        The number of clicks of the Start button. Not used.
    card_body : list
        The list of child entries in the current active tab.
    mode : str
        The id of the current active tab.
    start_ev : float
        The start eV value from the Store.
    end_ev : float
        The end eV value from the Store.
    ev_step : float
        The step eV value from the Store.
    time_step : float
        The time per step value from the Store.
    pass_no : int
        The no. of passes value from the Store.
    batch_dataframe : dict
        The dict with the batch mode data grid from the Store.

    Returns
    -------
    tuple[str, float, float, float, float, int, dict]
        Returns the current mode, start ev value, end ev value, step ev value, time per step value, no. of passes value, batch mode data grid, respectively, to the Store components.
    """
    if mode in "tab-2":
        dataframe = card_body[0]["props"]["children"][0]["props"]["data"]
        return (
            "batch",
            start_ev,
            end_ev,
            ev_step,
            time_step,
            pass_no,
            dataframe,
        )
    else:
        start_ev_mod = card_body[0]["props"]["children"][0]["props"]["children"][1]["props"]["value"]
        end_ev_mod = card_body[0]["props"]["children"][1]["props"]["children"][1]["props"]["value"]
        ev_step_mod = card_body[0]["props"]["children"][2]["props"]["children"][1]["props"]["value"]
        time_step_mod = card_body[0]["props"]["children"][3]["props"]["children"][1]["props"]["value"]
        pass_no_mod = card_body[0]["props"]["children"][4]["props"]["children"][1]["props"]["value"]
        return (
            "single",
            start_ev_mod,
            end_ev_mod,
            ev_step_mod,
            time_step_mod,
            pass_no_mod,
            batch_dataframe,
        )


# *********************************************************************************


@callback(
    Input("stop-click", "n_clicks"),
    prevent_initial_call=True,
)
def cancelOrStopMeasurement(n: int) -> None:
    """Method called when the Stop button is clicked on the app.

    Parameters
    ----------
    n : int
        Integer indicating the number of clicks of the stop-click button.

    """
    data_backend.interruptionClicked()


# ***********************************************************************************


@callback(
    Output(
        "check-running",
        "data",
        allow_duplicate=True,
    ),
    Output(
        "interval-component",
        "disabled",
        allow_duplicate=True,
    ),
    Output(
        "progress-interval",
        "disabled",
        allow_duplicate=True,
    ),
    Output(
        "graph",
        "figure",
        allow_duplicate=True,
    ),
    Input("start-mode", "data"),
    State("start-ev-value", "data"),
    State("end-ev-value", "data"),
    State("ev-step-value", "data"),
    State("time-step-value", "data"),
    State("meas-passes-value", "data"),
    State("batch-seq-dataframe", "data"),
    State("source-select", "value"),
    State("switch", "value"),
    prevent_initial_call=True,
)
def startMeasurement(
    mode: str,
    start_ev: float,
    end_ev: float,
    ev_step: float,
    time_step: float,
    pass_no: int,
    batch_dataframe: dict,
    source_mg: bool,
    switch_on: bool,
) -> tuple[bool, bool, bool, go.Figure]:
    """Method called after the initial preparation of the measurement was done.

    Parameters
    ----------
    mode : str
        String indicating if the mode of the measurement is single or batch mode.
    start_ev : float
        The start ev value to be used if a single mode measurement is selected.
    end_ev : float
        The end ev value to be used if a single mode measurement is selected.
    ev_step : float
        The step ev value to be used if a single mode measurement is selected.
    time_step : float
        The time per step value to be used if a single mode measurement is selected.
    pass_no : int
        The no. of passes value to be used if a single mode measurement is selected.
    batch_dataframe : dict
        The dictionary with the data from the batch data grid if the Batch mode measurement is selected.
    source_mg : bool
        Boolean indicating if the source of the X-Ray excitation is Mg.
    switch_on : bool
        Boolean indicating if the light mode theme is switched on.

    Returns
    -------
    tuple[bool, bool, bool, plotly.graph_objects.Figure]
        Boolean indicating if the measurement is running, boolean if the interval-component is to be disabled, boolean if the progress-interval is to be disabled, the plotly Figure object, respectively.
    """
    batch_sett = pd.DataFrame(batch_dataframe)
    dark_theme = False if switch_on else True
    if mode in "single":
        data_backend.startMeasurement(start_ev, end_ev, ev_step, time_step, pass_no, batch_sett, source_mg)
        fig = reset_fig(theme_dark=dark_theme)
    else:
        data_backend.startMeasurement(start_ev, end_ev, ev_step, time_step, pass_no, batch_sett, source_mg, True)
        fig = reset_fig(theme_dark=dark_theme, batch_mode=True)

    return True, False, False, fig


# ********************************************************************************


@callback(
    Output(
        "start-click",
        "disabled",
        allow_duplicate=True,
    ),
    Output(
        "stop-click",
        "disabled",
        allow_duplicate=True,
    ),
    Output(
        "shutdown-app",
        "disabled",
        allow_duplicate=True,
    ),
    Output("meas-select-card", "class_name"),
    Output(
        "interval-component",
        "disabled",
        allow_duplicate=True,
    ),
    Output(
        "save-results",
        "disabled",
        allow_duplicate=True,
    ),
    Output(
        "source-select",
        "disabled",
        allow_duplicate=True,
    ),
    Input("check-running", "data"),
    prevent_initial_call=True,
)
def checkRunningProgress(
    running: bool,
) -> tuple[bool, bool, bool, str, bool, bool, bool]:
    """Method called when the check-running boolean in the Store is updated.

    Parameters
    ----------
    running : bool
        Boolean indicating if the measurement is running.

    Returns
    -------
    tuple[bool, bool, bool, str, bool, bool, bool]
        The boolean indicating if the start button is disabled, boolean indicating if the stop button is disabled, boolean indicating if the shutdown button is disabled, the class name for the Card component, boolean indicating if the graph interval is disabled, boolean indicating if the save button is disabled, boolean indicating if the select source switch is disabled, respectively.
    """
    if running:
        return (
            True,  # disable the start button
            False,  # disable the stop button
            True,  # disable the shutdown button
            "mb-3 card-disable",  # class name for the Card component
            False,  # disable the graph interval component
            True,  # disable the save results button
            True,  # disable the source select switch
        )
    else:
        return (
            False,  # disable the start button
            True,  # disable the stop button
            False,  # disable the shutdown button
            "mb-3",  # class name for the Card component
            True if data_backend.meas_completed else False,  # disable the graph interval component
            False,  # disable the save results button
            False,  # disable the source select switch
        )


# ********************************************************************************


@callback(
    Output("progress-bar", "value"),
    Output("check-running", "data"),
    Output("kin-energy-disp", "value"),
    Output("binding-energy-disp", "value"),
    Output("time-elapsed-disp", "value"),
    Output("time-remain-disp", "value"),
    Input("progress-interval", "n_intervals"),
    prevent_initial_call=True,
)
def updateProgress(n: int) -> tuple[int, bool, float, float, float, float]:
    """Method called by the progress-interval interval component on firing.

    Parameters
    ----------
    n : int
        Integer indicating the number of intervals the interval component was fired.

    Returns
    -------
    tuple[int, bool, float, float, float, float]
        Integer for the current value of the progress bar, boolean indicating if the measurement is running, current kinetic energy value of the measurement, current binding energy value of the measurement, the elapsed time of the measurement, the remaining time of the measurement, respectively.
    """
    return (
        data_backend.current_progress,
        data_backend.meas_running,
        data_backend.current_kinetic_energy,
        data_backend.current_binding_energy,
        data_backend.elapsed_time,
        data_backend.remaining_time,
    )


# ********************************************************************************


@callback(
    Output(
        "graph",
        "figure",
        allow_duplicate=True,
    ),
    Output(
        "interval-component",
        "disabled",
        allow_duplicate=True,
    ),
    Input("interval-component", "n_intervals"),
    prevent_initial_call=True,
)
def updateGraphLive(n: int) -> tuple[Patch, bool]:
    """Method that is called in periodic intervals.

    Parameters
    ----------
    n : int
        The iteration of the update interval.

    Returns
    -------
    tuple[dash.Patch, bool]
        The patch for the figure object and the boolean indicating if the interval-component is to be disabled.
    """
    patched_figure = Patch()
    old_fig = data_backend.plot_fig
    data = old_fig.to_dict()["data"]
    patched_figure["data"] = data

    return patched_figure, not data_backend.meas_running


# *********************************************************************************


@callback(
    Output(
        "modal",
        "is_open",
        allow_duplicate=True,
    ),
    Input("save-results", "n_clicks"),
    Input("cancel-save", "n_clicks"),
    State("modal", "is_open"),
    prevent_initial_call=True,
)
def confirmFileSaveModalOpen(n1: int, n2: int, is_open: bool) -> bool:
    """Method called when the Save results button is clicked. It will open a Modal to confirm the file save.

    Parameters
    ----------
    n1 : int
        Integer holding the no. of clicks on the Save button.
    n2 : int
        Integer holding the number of clicks on the Cancel button in the Modal.
    is_open : bool
        Boolean indicating if the Modal is currently open.

    Returns
    -------
    bool
        Boolean indicating if the Modal is to be opened.
    """
    if n1 or n2:
        return not is_open
    return is_open


# *********************************************************************************


@callback(
    Output("confirm-save", "disabled"),
    Output("save-as-file", "valid"),
    Output("save-as-file", "invalid"),
    Input("save-as-file", "value"),
    prevent_initial_call=True,
)
def checkFilenameValidity(filename: str) -> tuple[bool, bool, bool]:
    """Method to check filename validity.

    Parameters
    ----------
    filename : str
        Name of the file to be saved as.

    Returns
    -------
    tuple[bool, bool, bool]
        Boolean to disabled the save button in the modal, enable valid property of the input field, enable invalid property of the input field respectively.
    """
    pattern = r'[^.\\/:*?"\'<>|]+'
    match = re.fullmatch(pattern, filename)
    if match is not None:
        return False, True, False
    else:
        return True, False, True
    
# ********************************************************************************  


@callback(
    Output(
        "modal",
        "is_open",
        allow_duplicate=True,
    ),
    Input("confirm-save", "n_clicks"),
    State("graph", "figure"),
    State("save-as-file", "value"),
    prevent_initial_call=True,
)
def saveDataAndPlot(n_clicks: int, ex_fig: dict, filename: str) -> bool:
    """Method called when Save Results button is clicked.

    Parameters
    ----------
    n_clicks : int
        Integer indicating the number of times the save-results button was clicked.
    ex_fig : dict
        The dictionary with the figure object of the graph area in the Dash app.
    filename : str
        The filename to save the results and plot as.

    Returns
    -------
    bool
        The boolean to close the Modal.
    """
    old_fig = go.Figure(ex_fig)
    old_fig.write_html(
        filename + ".html",
        config=CONFIG,
        include_plotlyjs="cdn",
        include_mathjax="cdn",
    )
    data_backend.saveMeasurementData(filename)
    return False


# *********************************************************************************


@callback(
    Input("shutdown-app", "n_clicks"),
    prevent_initial_call=True,
)
def shutdownApp(n: int) -> None:
    """Method called when the shutdown-app button is clicked.

    Parameters
    ----------
    n : int
        Integer indicating the number of times the shutdown-app button was clicked.

    """
    data_backend.onClose()


# *********************************************************************************


if __name__ == "__main__":
    # app.run(debug=True, port="8060")
    # app.server.http_server.serve_forever()
    waitress.serve(app.server, host="localhost", port="8060", expose_tracebacks=True, threads=8)
