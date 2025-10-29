import sys
import warnings
import webbrowser
from collections import defaultdict
from collections.abc import Callable, Generator
from functools import partial
from itertools import combinations, cycle
from math import pi
from typing import Any

import bokeh
import numpy as np
import pandas as pd
from bokeh.colors import Color
from bokeh.colors.named import lime as BULL_COLOR
from bokeh.colors.named import tomato as BEAR_COLOR
from bokeh.embed import components
from bokeh.io import output_file, output_notebook, save, show
from bokeh.io.export import export_png
from bokeh.io.state import curstate
from bokeh.layouts import column, gridplot, row
from bokeh.models import (
    BasicTicker,
    CategoricalColorMapper,
    ColorBar,
    ColumnDataSource,
    CrosshairTool,
    CustomJS,
    CustomJSTransform,
    DataTable,
    DatetimeTickFormatter,
    HoverTool,
    HTMLTemplateFormatter,
    Label,
    LabelSet,
    LinearColorMapper,
    NumeralTickFormatter,
    PrintfTickFormatter,
    Range1d,
    TableColumn,
    WheelZoomTool,
)

try:
    from bokeh.models import CustomJSTickFormatter
except ImportError:
    from bokeh.models import FuncTickFormatter as CustomJSTickFormatter

from bokeh.palettes import Category10, RdYlGn
from bokeh.plotting import curdoc
from bokeh.plotting import figure as _figure
from bokeh.resources import CDN
from bokeh.transform import factor_cmap, transform
from jinja2 import Template

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

OHLC_RESAMPLE_RULE = {
    "open": "first",
    "close": "last",
    "high": "max",
    "low": "min",
    "volume": "sum",
    "turnover": "last",
}


def resample_agg_gen(columns: list[str]) -> dict[str, str]:
    return {col: OHLC_RESAMPLE_RULE.get(col, "last") for col in columns}


"""
TODO:
https://github.com/ranaroussi/quantstats
month return vs benchmark return
distribution of returns - normal distribution line & average
rolling beta(6mth & 12mth) & average
rolling volatility(6mth) & average
rolling sharpe(6mth)  & average
rolling sortino(6mth)  & average
top5 drawdown periods & top 10 in table
underwater plot - rolling max drawdown & average
return quantiles

"""

HTML_TEMPLATE = """
    <!DOCTYPE html>
        <html>
        <head>
            <title>BACKTEST REPORT</title>
            <meta charset="utf-8">
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/semantic-ui@2.4.2/dist/semantic.min.css">
            <script src="https://code.jquery.com/jquery-3.1.1.min.js" crossorigin="anonymous"></script>
            {{ resources }}
            {{ script }}
            <style>
                img {
                    filter: rgb(0, 0, 0.35);
                    /* IE6-9 */
                    -webkit-filter: grayscale(1);
                    /* Google Chrome, Safari 6+ & Opera 15+ */
                    filter: grayscale(1);
                    /* Microsoft Edge and Firefox 35+ */
                }
                /* Disable grayscale on hover */
                img:hover {
                    -webkit-filter: grayscale(0);
                    filter: none;
                }
            </style>
        </head>
        <body>
            <div class="ui container" style="height: 5%">
                {{ table }}
            </div>
            <div class="ui container">
                {{ div }}
            </div>
            </br>
            <div class="ui last center container">
                    <div style="color: rgba(0, 0, 0, 0.35);
                        font-size: 1rem;
                        font-weight: bold;
                        text-transform: uppercase;
                        font-family: Tahoma, sans-serif;
                        text-align: center;
                        <!--  transform: rotate(-45deg);  -->
                        user-select: none;">
                        <img width="20" height="20" src="https://areix-ai.com/wp-content/uploads/2019/09/Asset-1-1-e1590331845541.png" alt="">
                        <p>2025 @ Charon</p>
                    </div>
            </div>
        </div>
        </body>
    </html>
"""

template = Template(HTML_TEMPLATE)
_AUTOSCALE_JS_CALLBACK = """
    if (!window._bt_scale_range) {
        window._bt_scale_range = function (range, min, max, pad) {
            "use strict";
            if (min !== Infinity && max !== -Infinity) {
                pad = pad ? (max - min) * .03 : 0;
                range.start = min - pad;
                range.end = max + pad;
            } else console.error('backtesting: scale range error:', min, max, range);
        };
    }
    clearTimeout(window._bt_autoscale_timeout);
    window._bt_autoscale_timeout = setTimeout(function () {
        /**
        * @variable cb_obj `fig_ohlc.x_range`.
        * @variable source `ColumnDataSource`
        * @variable ohlc_range `fig_ohlc.y_range`.
        * @variable volume_range `fig_volume.y_range`.
        */
        "use strict";
        // console.log(cb_obj,'-----------cb_obj---------')
        // console.log(source,'-----------source---------')
        let i = Math.max(Math.floor(cb_obj.start), 0),
            j = Math.min(Math.ceil(cb_obj.end), source.data['ohlc_high'].length);
            // console.log(i,j,'-----------i,j---------')
        let max = Math.max.apply(null, source.data['ohlc_high'].slice(i, j)),
            min = Math.min.apply(null, source.data['ohlc_low'].slice(i, j));
        _bt_scale_range(ohlc_range, min, max, true);
        // console.log(ohlc_range,'---------ohlc_range-----------')
        // console.log(volume_range,'---------volume_range-----------')
        if (volume_range) {
            max = Math.max.apply(null, source.data['volume'].slice(i, j));
            _bt_scale_range(volume_range, 0, max * 1.03, false);
        }
    }, 50);
"""

STATS_TABLE_TEMPLATE = """
            <div>
            <%= (function textformatter(){
                    // money with dollor sign and comma
                    if(['standardized_expected_value','gross_winning_trades_amount', 'gross_losing_trades_amount','avg_winning_trades_amount','avg_losing_trades_amount','gross_trades_profit','gross_trades_loss','avg_winning_trades_pnl','avg_losing_trades_pnl','largest_profit_winning_trade','largest_loss_losing_trade','expected_value','net_investment','holding_values','var','beginning_balance','capital','ending_balance','available_balance','total_net_pnl', 'total_commission','total_turnover', 'gross_profit', 'gross_loss', 'max_win_in_day', 'max_loss_in_day', 'avg_profit_per_trade($)', 'avg_amount_per_trade', 'avg_amount_per_open_trade', 'avg_amount_per_closed_trade', 'avg_amount_per_long_trade', 'avg_amount_per_short_trade','avg_pnl_per_trade($)', 'avg_daily_pnl($)', 'avg_weekly_pnl($)', 'avg_monthly_pnl($)', 'avg_quarterly_pnl($)', 'avg_annualy_pnl($)','past_24hr_pnl'].includes(index)){

                        return "$"+(value).toLocaleString() //.toFixed(1)
                    }
                    // two decimal place number
                    else if(['common_sense_ratio','skew','kurtosis','profit_factor','avg_daily_trades','avg_weekly_trades', 'avg_monthly_trades', 'payoff_ratio','sharpe_ratio', 'sortino_ratio', 'annualized_volatility', 'omega_ratio', 'downside_risk', 'information_ratio', 'beta', 'alpha', 'calmar_ratio', 'tail_ratio','stability_of_timeseries','sqn', 'avg_daily_risk_score', 'avg_risk_score_past_7days','risk_score'].includes(index)){

                        return Number(value).toFixed(3)
                    }
                    // percentage with two decimal places
                    else if(['pct_time_in_market','past_7d_roi','past_14d_roi','past_30d_roi','past_90d_roi','past_180d_roi','past_1yr_roi','total_return', 'return_on_capital', 'max_return','min_return', 'return_on_investment', 'return_on_initial_capital','annualized_return','past_24hr_roi','past_24hr_apr','win_ratio','loss_ratio' ,'win_ratio_long','loss_ratio_long' ,'win_ratio_short','loss_ratio_short' ,'avg_pnl_per_trade','avg_profit_per_trade', 'avg_daily_pnl', 'avg_weekly_pnl', 'avg_monthly_pnl', 'avg_quarterly_pnl', 'avg_annualy_pnl', 'max_drawdown', 'max_runup'].includes(index)){
                        return (value*100).toFixed(2) +'%'
                    }
                    // period
                    else if(['max_drawdown_duration','max_runup_duration','duration'].includes(index)){
                        t = value / 1000;
                        var d = Math.floor(t/86400),
                            h = ('0'+Math.floor(t/3600) % 24).slice(-2),
                            m = ('0'+Math.floor(t/60)%60).slice(-2),
                            s = ('0' + t % 60).slice(-2);
                        return (d>0?d+'d ':'')+(h>0?h+':':'')+(m>0?m+':':'')+(t>60?s:s+'s');
                    }
                    // datetime
                    else if(['start', 'end'].includes(index)){
                        date = new Date(value+ 8 * 60 * 60 * 1000); // add 8 hours in milliseconds
                        //date_s = date.toLocaleDateString("en-US")
                        date_s = date.toISOString().replace("T", " ").slice(0,-5)
                        //date_s = date.toString("yyyy-MM-dd HH:mm:ss")
                        console.log('after:', index, value, date_s, 'date.toISOString:', date.toISOString())
                        return date_s + ' (UTC+8)'
                    }
                    else if (["max_drawdown_period", "max_runup_period"].includes(index)){
                        var tmp = [];
                        value.forEach(d => {
                            date = new Date(d);
                            date_s = date.toISOString().replace("T", " ").slice(0,-5)
                            tmp.push(date_s)
                        })
                        return tmp
                    }
                    else{
                        // console.log("=============>", index, value,'|||||||',typeof value)
                        return (value)
                    }
                }()) %></div>
            """

# IS_JUPYTER_NOTEBOOK = "JPY_PARENT_PID" in os.environ
IS_JUPYTER_NOTEBOOK = "ipykernel" in sys.modules
if IS_JUPYTER_NOTEBOOK:
    output_notebook()


def set_bokeh_output(notebook: bool = False) -> None:
    global IS_JUPYTER_NOTEBOOK
    IS_JUPYTER_NOTEBOOK = notebook


def _bokeh_reset(filename: str | None = None) -> None:
    curstate().reset()
    if filename:
        if not filename.endswith(".html"):
            filename += ".html"
        output_file(filename, title=filename)
    elif IS_JUPYTER_NOTEBOOK:
        curstate().output_notebook()


def _bokeh_reset_html(html: str, filename: str | None = None) -> None:
    # print(f"   >>> _bokeh_reset_html - filename:{filename} | IS_JUPYTER_NOTEBOOK:{IS_JUPYTER_NOTEBOOK}")
    curstate().reset()
    if filename:
        if not filename.endswith(".html"):
            filename += ".html"
        # print(filename,'!!!')
        with open(filename, mode="w") as f:
            f.write(html)
    if IS_JUPYTER_NOTEBOOK:
        curstate().output_notebook()


def colorgen() -> Generator[str, None, None]:
    yield from cycle(Category10[10])


def lightness(color: Color, lightness: float = 0.94) -> Color:
    color = color.to_rgb()
    color.l = lightness
    return color.to_rgb()


_MAX_CANDLES = 15_000
# _MAX_CANDLES = 300
COLORS = [BEAR_COLOR, BULL_COLOR]
COLORS_DARKER = [lightness(BULL_COLOR, 0.35), lightness(BEAR_COLOR, 0.35)]
BAR_WIDTH = 0.8


class Plotter:
    def __init__(
        self,
        feeds: list[Any] = [],
        portfolio_data: pd.DataFrame = pd.DataFrame(),
        stats: Any = None,
        all_trades: list[dict[str, Any]] = [],
        resample_rule: str | None = None,
        custom_plot_data: dict[str, Any] = {},
        draw_figure: Callable[[], Any] | None = None,
        in_nav: bool = False,
        nav_stats: Any = None,
    ) -> None:
        self.draw_figure = draw_figure() if draw_figure else None
        self.resample_rule = resample_rule
        self.custom_plot = custom_plot_data  # self.custom_plot[figure_key][graph_key][self.tick] = data_point
        self.feeds = feeds
        self.stats = stats
        self.portfolio_data = portfolio_data
        self.trade_records = defaultdict(list)
        for trade in all_trades:
            self.trade_records[trade["aio_symbol"]].append(trade)

        self.doc = curdoc()
        self.in_nav = in_nav
        self.nav_stats = nav_stats

    def plot(self, path: str, interactive: bool = False, **kwargs: Any) -> Any:
        if interactive:
            self.plot_interactive(path, **kwargs)
        else:
            self.plot_interactive(path, export_as_png=True, **kwargs)

    def plot_static(self, path: str) -> None:
        pass

    def plot_interactive(
        self,
        path: str,
        plot_width: int | None = None,
        plot_candlestick: bool = True,
        plot_equity: bool = True,
        plot_return: bool = False,
        plot_pnl: bool = True,
        plot_volume: bool = True,
        plot_drawdown: bool = False,
        smooth_equity: bool = False,
        resample: bool = True,
        show_legend: bool = True,
        open_browser: bool | None = None,
        export_as_png: bool = False,
        extra_ohlc: bool = False,
        only_ohlc: bool = False,
        zoom_tgt: bool = True,
        **kwargs: Any,
    ) -> Any:
        """
        1. in ipynb - plot,
            if not open browser, show
        2. in script - plot,
            if not open browser, show
        3. in script - content_oputput
            if not open browser

        """
        # if not path and open_browser:
        #     path = '.'
        # print(f">>> 1path:{path}; open_browser:{open_browser}; IS_JUPYTER_NOTEBOOK:{IS_JUPYTER_NOTEBOOK}")
        # if open_browser is None:
        #     open_browser = not IS_JUPYTER_NOTEBOOK

        if not path and not IS_JUPYTER_NOTEBOOK:
            # print(f'................ IS_JUPYTER_NOTEBOOK:{IS_JUPYTER_NOTEBOOK}')
            return
        # print(f">>> 2path:{path}; open_browser:{open_browser}; IS_JUPYTER_NOTEBOOK:{IS_JUPYTER_NOTEBOOK}")
        try:
            width = 1200 if export_as_png else plot_width
            # data = self.stat.data.copy()
            data = self.portfolio_data
            # COLUMNS: available_balance  holding_value   assets_value  position_count  total_commission  total_turnover  total_realized_pnl  total_unrealized_pnl  total_net_investment     total_pnl  positions  pct_returns     pnl_net    pnl_cumsum  benchmark_pct_returns  benchmark  datetime

            # print(data,'aiiiiiiiiiiiiii', data.index, data.columns)
            if "positions" in data.columns:
                data.drop(["positions"], axis=1, inplace=True)

            stats = self.nav_stats if self.in_nav else self.stats
            # top10_symbols = []
            # # current_weight = [{'symbol':pos['symbol'], 'weight': pos['weight_wo_cash']} for pos in stats['positions'] ]
            # current_weight = [{"symbol": pos["symbol"], "weight": pos["weight_wo_cash"]} for pos in stats["positions"]]
            # current_weight_df = pd.DataFrame(current_weight)
            # if not current_weight_df.empty:
            #     current_weight_df["weight"] = round(current_weight_df["weight"] * 100, 2)
            #     current_weight_df = current_weight_df.sort_values(["weight"], ascending=False)
            #     current_weight_df = current_weight_df.head(10)  # TODO: top9 + rest in others
            #     top10_symbols = current_weight_df["symbol"].unique().tolist()
            # # print(f"top10_symbols: {top10_symbols} init")

            data.index.name = None
            # data["datetime"] = data.index
            data["datetime_str"] = data.datetime.apply(lambda x: x.strftime("%a %d %b %Y %H:%M:%S %p %Z"))
            data["datetime_index"] = data.index
            data["datetime_ntz"] = data.index.tz_localize(None)

            data = data.reset_index(drop=True)  # index become sequence of numbers
            index = data.index

            # print(f"---> plotter stats data: {data}; index: {index}")

            # print(index,'???? X-AXIS ??????') # RangeIndex(start=0, stop=243, step=1)
            source = ColumnDataSource(data)
            source.add((data["pnl_net"] >= 0).values.astype(np.uint8).astype(str), "inc")
            inc_cmap = factor_cmap("inc", COLORS, ["0", "1"])
            ohlc_inc_cmap = factor_cmap("ohlc_inc", COLORS, ["0", "1"])
            colors = colorgen()
            new_bokeh_figure = partial(
                _figure,
                height=400,
                tools="xpan,xwheel_zoom, ywheel_zoom, box_zoom,undo,redo,reset,save",
                active_drag="xpan",
                active_scroll="xwheel_zoom",
            )
            # new_bokeh_figure = partial(_figure, x_axis_type="linear", width=width, height=400, tools="xpan,xwheel_zoom,ywheel_zoom,box_zoom,undo,redo,reset,save", active_drag="xpan", active_scroll="xwheel_zoom")
            figs_above_ohlc, fig_ohlcs, figs_below_ohlc = [], [], []
            fig_ohlcs_only = []
            # if index.empty:
            #     return
            # else:
            pad = (index[-1] - index[0]) / 20 if index.size > 1 else None
            _xrange = (
                Range1d(
                    index[0],
                    index[-1],
                    min_interval=10,
                    bounds=(index[0] - pad, index[-1] + pad),
                )
                if index.size > 1
                else None
            )
            _xaxis_formatter = CustomJSTickFormatter(
                args=dict(
                    # formatter=DatetimeTickFormatter(days=["%d %b", "%a %d"], months=["%m/%Y", "%b'%y"]),
                    # formatter=DatetimeTickFormatter(hourmin="%H:%M", hours="%Hh", minutes=":%M", days="%d %b", months="%m/%Y"),
                    formatter=DatetimeTickFormatter(
                        hourmin="%H:%M",
                        hours="%H:%M",
                        minutes="%H:%M",
                        days="%d %b",
                        months="%m/%Y",
                    ),
                    source=source,
                ),
                code="""
            this.labels = this.labels || formatter.doFormat(ticks
                                                            .map(i => source.data.datetime_ntz[i])
                                                            .filter(t => t !== undefined));
            return this.labels[index] || "";
                    """,
            )

            def new_indicator_figure(
                x_range: Range1d | None = None, x_axis_type: str = "linear", **kwargs: Any
            ) -> _figure:
                kwargs.setdefault("height", 90)
                ### NOTE: share same x_range will zoom together
                x_range = (
                    x_range
                    if x_range
                    else (
                        _xrange
                        if zoom_tgt
                        else Range1d(
                            index[0],
                            index[-1],
                            min_interval=10,
                            bounds=(index[0] - pad, index[-1] + pad),
                        )
                        if index.size > 1
                        else None
                    )
                )
                if x_axis_type is None:
                    if x_range is None:
                        fig = new_bokeh_figure(active_scroll="xwheel_zoom", active_drag="xpan", **kwargs)
                    else:
                        fig = new_bokeh_figure(
                            x_range=x_range,
                            active_scroll="xwheel_zoom",
                            active_drag="xpan",
                            **kwargs,
                        )
                elif x_range is None:
                    fig = new_bokeh_figure(
                        x_axis_type=x_axis_type,
                        active_scroll="xwheel_zoom",
                        active_drag="xpan",
                        **kwargs,
                    )
                else:
                    fig = new_bokeh_figure(
                        x_range=x_range,
                        x_axis_type=x_axis_type,
                        active_scroll="xwheel_zoom",
                        active_drag="xpan",
                        **kwargs,
                    )
                fig.xaxis.visible = False
                fig.yaxis.minor_tick_line_color = None
                return fig

            def set_tooltips(
                fig: _figure,
                tooltips: tuple[tuple[str, str], ...] = (),
                vline: bool = True,
                renderers: tuple[Any, ...] = (),
                include_dt: bool = True,
            ) -> None:
                tooltips = list(tooltips)
                renderers = list(renderers)
                if include_dt:
                    formatters = {"@datetime": "datetime"}
                    # NBSP = "\N{NBSP}" * 4
                    # tooltips = [("Datetime", "@datetime_str"), ("x, y", NBSP.join(("$index", "$y{0,0.0[0000]}")))] + tooltips
                    tooltips = [("Datetime", "@datetime_str")] + tooltips

                    fig.add_tools(
                        HoverTool(
                            point_policy="follow_mouse",
                            renderers=renderers,
                            formatters=formatters,
                            tooltips=tooltips,
                            mode="vline" if vline else "mouse",
                            toggleable=False,
                        )
                    )
                else:
                    fig.add_tools(
                        HoverTool(
                            point_policy="follow_mouse",
                            renderers=renderers,
                            tooltips=tooltips,
                            toggleable=False,
                        )
                    )

            def _plot_custom_supplement(
                graph_data: dict[str, Any], fig: _figure, source: ColumnDataSource, tooltips: list[tuple[str, str]]
            ) -> tuple[Any, list[tuple[str, str]]]:
                try:
                    for grpah_info, dps in graph_data.items():
                        raw = grpah_info.split("|")
                        grpah_name = raw[0]
                        grpah_type = raw[1]
                        grpah_color = raw[2] if raw[2] else next(colors)
                        # figure_height = int(raw[3])
                        # print(dps)
                        series = pd.Series(dps, name=grpah_name)
                        series.index = pd.to_datetime(series.index)
                        # print(series,'???????', series.index.dtype,'!!!',data["datetime"],'/n===================')
                        series = (
                            data["datetime"].to_frame().merge(series, left_on="datetime", right_index=True, how="left")
                        )
                        series = series.fillna(method="ffill").fillna(method="bfill")
                        # print(series,'!!!!!!!!!!!')
                        source.add(series[grpah_name], grpah_name)
                        tooltip_format = "@{" + grpah_name + "}{0,0.[000]}"
                        if grpah_type == "bar":
                            source.add(
                                (series[grpah_name] >= 0).values.astype(np.uint8).astype(str),
                                "cus_inc",
                            )
                            cus_inc_cmap = factor_cmap("cus_inc", COLORS, ["0", "1"])

                            r = fig.vbar(
                                "index",
                                BAR_WIDTH,
                                grpah_name,
                                source=source,
                                color=grpah_color if raw[2] else cus_inc_cmap,
                                legend_label=grpah_name,
                            )
                        elif grpah_type == "scatter":
                            r = fig.scatter(
                                x="index",
                                y=grpah_name,
                                alpha=0.7,
                                color=grpah_color,
                                line_color="black",
                                size=12,
                                source=source,
                                legend_label=grpah_name,
                            )
                        else:
                            r = fig.line(
                                "index",
                                grpah_name,
                                source=source,
                                line_width=1.5,
                                line_alpha=1,
                                line_color=grpah_color,
                                legend_label=grpah_name,
                            )
                        # NBSP = "\N{NBSP}" * 4
                        # tooltips.append(("x, y", NBSP.join(("$index", "$y{0,0.0[0000]}"))))
                        tooltips.append((grpah_name, tooltip_format))
                    return r, tooltips
                except Exception as e:
                    tb_lineno = e.__traceback__.tb_lineno if e.__traceback__ else "unknown"
                    tb_filename = (
                        e.__traceback__.tb_frame.f_code.co_filename
                        if e.__traceback__ and e.__traceback__.tb_frame
                        else "unknown"
                    )
                    print(
                        f"Something went wrong when _plot_custom_supplement. Error:{str(e)} in line {tb_lineno} for file {tb_filename}."
                    )
                    return None, None

            def _plot_custom(figure_name: str, graph_data: dict[str, Any], height: float = 110) -> _figure:
                """Custom section"""
                try:
                    # print(f"[Debug - figure_name:{figure_name}] | graph_data name:{list(graph_data.keys())[0]}; graph_data length:{len(list(graph_data.values())[0])} || ({len(data['datetime'])})")
                    yaxis_label = figure_name
                    # print(f">>>>>figure_name:{figure_name} ||  graph_data:{graph_data}")
                    fig = new_indicator_figure(y_axis_label=yaxis_label, **(dict(height=height)))
                    fig.xaxis.formatter = _xaxis_formatter
                    fig.xaxis.visible = True
                    tick_format = "0,0.[00]"
                    # legend_format = "{:,.0f}"

                    fig.yaxis.formatter = NumeralTickFormatter(format=tick_format)
                    tooltips: list[tuple[str, str]] = []
                    r, tooltips = _plot_custom_supplement(graph_data, fig, source, tooltips)
                    if r:
                        set_tooltips(fig, tooltips, renderers=[r])

                    return fig
                except Exception as e:
                    tb_lineno = e.__traceback__.tb_lineno if e.__traceback__ else "unknown"
                    tb_filename = (
                        e.__traceback__.tb_frame.f_code.co_filename
                        if e.__traceback__ and e.__traceback__.tb_frame
                        else "unknown"
                    )
                    print(
                        f"Something went wrong when _plot_custom. Error:{str(e)} in line {tb_lineno} for file {tb_filename}."
                    )
                    return None

            def _plot_return_section(
                is_return: bool = False,
                relative_equity: bool = False,
                xaxis_visible: bool = False,
                height: float = 145,
            ):
                try:
                    equity = data["assets_value"].copy()
                    if "max_drawdown" in stats.index and stats.loc["max_drawdown"]:
                        max_dd = float(stats.loc["max_drawdown"])
                        dd_start, dd_end = stats.loc["max_drawdown_period"]
                        dd_timedelta_label = dd_end - dd_start
                        # print(dd_start, dd_end ,'!!!!', data)
                        # dd_start, dd_end = (
                        #     data[data["datetime"] == dd_start].index[0],
                        #     data[data["datetime"] == dd_end].index[0],
                        # )
                        dd_start, dd_end = (
                            data[data["datetime_index"] == dd_start].index[0],
                            data[data["datetime_index"] == dd_end].index[0],
                        )
                        if np.isnan(dd_end):
                            dd_start = dd_end = equity.index[0]
                        else:
                            dd_start = equity[:dd_end].idxmax()
                            if dd_end != equity.index[-1]:
                                dd_end = np.interp(
                                    equity[dd_start],
                                    (equity[dd_end - 1], equity[dd_end]),
                                    (dd_end - 1, dd_end),
                                )
                                dd_end = int(dd_end)
                    if "max_runup" in stats.index and stats.loc["max_runup"]:
                        max_ru = float(stats.loc["max_runup"])
                        ru_start, ru_end = stats.loc["max_runup_period"]
                        ru_timedelta_label = ru_end - ru_start
                        ru_start, ru_end = (
                            data[data["datetime_index"] == ru_start].index[0],
                            data[data["datetime_index"] == ru_end].index[0],
                        )
                        if np.isnan(ru_end):
                            ru_start = ru_end = equity.index[0]
                        else:
                            ru_start = equity[:ru_end].idxmin()
                            if ru_end != equity.index[-1]:
                                ru_end = np.interp(
                                    equity[ru_start],
                                    (equity[ru_end - 1], equity[ru_end]),
                                    (ru_end - 1, ru_end),
                                )
                                ru_end = int(ru_end)
                    yaxis_label = "Return" if is_return else "Portfolio Value"
                    source_key = "return" if is_return else "portfolio_value"
                    fig = new_indicator_figure(y_axis_label=yaxis_label, **(dict(height=height)))

                    if equity.empty:
                        return fig

                    if relative_equity:
                        equity /= equity.iloc[0]
                    if is_return:
                        equity -= equity.iloc[0]
                    source.add(equity, source_key)

                    if relative_equity:
                        tooltip_format = f"@{source_key}{{+0,0.[000]%}}"
                        tick_format = "0,0.[00]%"
                        legend_format = "{:,.2f}%"
                        if "benchmark" in data.columns:
                            v_func = """
                                //console.log(peer, xs);
                                var res = []
                                for (var i = 0; i < xs.length; i++) {
                                    res.push(peer[i] > xs[i] ? peer[i] : xs[i])
                                }
                                return res
                            """
                            fig.varea(
                                x="index",
                                y1=transform(
                                    source_key,
                                    CustomJSTransform(
                                        args=dict(peer=source.data["benchmark"]),
                                        v_func=v_func,
                                    ),
                                ),
                                y2="benchmark",
                                source=source,
                                color="palegreen",
                                fill_alpha=0.45,
                            )
                            fig.varea(
                                x="index",
                                y1=transform(
                                    "benchmark",
                                    CustomJSTransform(
                                        args=dict(peer=source.data[source_key]),
                                        v_func=v_func,
                                    ),
                                ),
                                y2=source_key,
                                source=source,
                                color="lightsalmon",
                                fill_alpha=0.45,
                            )
                            fig.line(
                                "index",
                                "benchmark",
                                source=source,
                                line_color="black",
                                line_width=1.5,
                                line_alpha=1,
                                legend_label=f"Benchmark - {stats.loc['benchmark']}",
                            )
                    else:
                        tooltip_format = f"@{source_key}{{$ 0,0}}"
                        # tooltip_format = f"@{source_key}{{$ 0,0.[000]}}"
                        tick_format = "$ 0.0 a"
                        legend_format = "${:,.0f}"

                        ### DEBUG: Bug at 20231114 version 2.4.3
                        patch_index = np.r_[index, index[::-1]]
                        patch_equity_dd = np.r_[equity, equity.cummax()[::-1]]
                        # print('>>>>1 index',patch_index, len(patch_index))
                        # print('>>>>2 equity_dd',patch_equity_dd, len(patch_equity_dd),)
                        if len(patch_equity_dd) < _MAX_CANDLES * 2:
                            fig.patch(
                                "index",
                                "equity_dd",
                                source=ColumnDataSource(dict(index=patch_index, equity_dd=patch_equity_dd)),
                                fill_color="#ffffea",
                                line_color="#ffcb66",
                            )

                    r = fig.line("index", source_key, source=source, line_width=1.5, line_alpha=1)
                    tooltips = (
                        [
                            (yaxis_label, tooltip_format),
                            ("Benchmark", "@benchmark{+0,0.[000]%}"),
                        ]
                        if "benchmark" in data.columns
                        else [(yaxis_label, tooltip_format)]
                    )
                    set_tooltips(fig, tooltips, renderers=[r])
                    fig.yaxis.formatter = NumeralTickFormatter(format=tick_format)
                    # print('!!!',equity, equity.empty, equity.idxmax(),'!!!')
                    if equity.empty:
                        return fig
                    equity = equity.fillna(0.0)
                    argmax = equity.idxmax()
                    argmin = equity.idxmin()
                    # if not np.isnan(argmax):
                    fig.scatter(
                        argmax,
                        equity[argmax],
                        legend_label=f"Peak ({legend_format.format(equity[argmax] * (100 if relative_equity else 1))})",
                        color="cyan",
                        size=8,
                    )
                    fig.scatter(
                        argmin,
                        equity[argmin],
                        legend_label=f"Bottom ({legend_format.format(equity[argmin] * (100 if relative_equity else 1))})",
                        color="grey",
                        size=8,
                    )
                    fig.scatter(
                        index[-1],
                        equity.values[-1],
                        legend_label=f"Final ({legend_format.format(equity.iloc[-1] * (100 if relative_equity else 1))})",
                        color="blue",
                        size=8,
                    )
                    if "max_drawdown" in stats.index and stats.loc["max_drawdown"]:
                        fig.scatter(
                            dd_end,
                            equity.iloc[dd_end],
                            legend_label=f"Max Drawdown (-{100 * max_dd:.1f}%)",
                            color="red",
                            size=8,
                        )
                        lb = (
                            f"Max Dd Dur. ({dd_timedelta_label})".replace(", 0:00:00", "")
                            .replace(" 00:00:00", "")
                            .replace("(0 days ", "(")
                        )
                        fig.line(
                            [dd_start, dd_end],
                            [equity.iloc[dd_start], equity.iloc[dd_end]],
                            line_color="red",
                            line_width=2,
                            legend_label=lb,
                        )
                    if "max_runup" in stats.index and stats.loc["max_runup"]:
                        fig.scatter(
                            ru_end,
                            equity.iloc[ru_end],
                            legend_label=f"Max Runup (+{100 * max_ru:.1f}%)",
                            color="green",
                            size=8,
                        )
                        lb = (
                            f"Max Ru Dur. ({ru_timedelta_label})".replace(", 0:00:00", "")
                            .replace(" 00:00:00", "")
                            .replace("(0 days ", "(")
                        )
                        fig.line(
                            [ru_start, ru_end],
                            [equity.iloc[ru_start], equity.iloc[ru_end]],
                            line_color="green",
                            line_width=2,
                            legend_label=lb,
                        )
                    if xaxis_visible:
                        fig.xaxis.formatter = _xaxis_formatter
                        fig.xaxis.visible = True
                    return fig
                except Exception as e:
                    tb_lineno = e.__traceback__.tb_lineno if e.__traceback__ else "unknown"
                    tb_filename = (
                        e.__traceback__.tb_frame.f_code.co_filename
                        if e.__traceback__ and e.__traceback__.tb_frame
                        else "unknown"
                    )
                    print(
                        f"Something went wrong when _plot_return_section. Error:{str(e)} in line {tb_lineno} for file {tb_filename}."
                    )
                    return None

            def _plot_drawdown_section(height: float = 185):
                try:
                    fig = new_indicator_figure(y_axis_label="Drawdown", **(dict(height=height)))
                    drawdown_series = stats.loc["drawdown_series"]

                    df = pd.DataFrame.from_dict(drawdown_series, orient="index", columns=["dd"])  #
                    df.index = pd.to_datetime(df.index)
                    if df.index.tz:
                        df.index = df.index.tz_convert("Asia/Hong_Kong")
                    else:
                        df.index = df.index.tz_localize("Asia/Hong_Kong")
                    # print(f'~~~~~~~~~ {df.tail(5)} ~~~~~~~~~1')
                    # print(f'~~~~~~~~~ { df.index} ({ df.index.dtype}) ~~~~~~~~~1')

                    df = df.loc[
                        df.index.isin(data["datetime"])
                    ].copy()  ### filter the datafeed datetime that is in the return data
                    df.index.name = None
                    df["datetime"] = df.index
                    df["datetime_str"] = df.datetime.apply(lambda x: x.strftime("%a %d %b %Y %H:%M:%S %p %Z"))
                    df = df.reset_index(
                        drop=True
                    )  # FIXME: x-axis's data in consist of index instead of using dt directly, so when reset_index, all the dt will start from 0
                    df.index = data.loc[
                        data.datetime.isin(df["datetime"])
                    ].index  # NOTE: This index match code fixed the above issue
                    # print(f'~~~~~~~~~ {df.tail(5)} ~~~~~~~~~')

                    # print('======> ',aio_symbol, df,'!!!!!', data["datetime"], data["datetime"].dtype)

                    if df.empty:
                        return fig

                    feed_source = ColumnDataSource(df)

                    # equity = equity.fillna(0.0)
                    # argmax = equity.idxmax()
                    # argmin = equity.idxmin()
                    # # if not np.isnan(argmax):
                    # fig.scatter(argmax, equity[argmax], legend_label="Peak ({})".format(legend_format.format(equity[argmax] )), color="cyan", size=8)
                    # fig.scatter(argmin, equity[argmin], legend_label="Bottom ({})".format(legend_format.format(equity[argmin] )), color="grey", size=8)
                    # fig.scatter(index[-1], equity.values[-1], legend_label="Final ({})".format(legend_format.format(equity.iloc[-1] )), color="blue", size=8)

                    r = fig.line(
                        "index",
                        "dd",
                        source=feed_source,
                        line_color="grey",
                        line_width=1.5,
                        line_alpha=1,
                        legend_label="drawdown",
                    )
                    tooltip_format = "@dd{+0,0.[000]%}"
                    tick_format = "0,0.[00]%"
                    # legend_format = "{:,.2f}%"
                    tooltips = [("Max Drawdown", tooltip_format)]
                    set_tooltips(fig, tooltips, renderers=[r])
                    fig.yaxis.formatter = NumeralTickFormatter(format=tick_format)

                    fig.xaxis.formatter = _xaxis_formatter
                    fig.xaxis.visible = True

                    # fig.patch("index", "equity_dd", source=ColumnDataSource(dict(index=np.r_[index, index[::-1]], equity_dd=np.r_[df['dd'], df['dd'].cummax()[::-1]])), fill_color="#ffffea", line_color="#ffcb66")

                    min_value = df["dd"].min()
                    max_value = df["dd"].max()
                    min_value_idx = df["dd"].idxmin()
                    # print(f">>> min_value_idx:{min_value_idx}; min_value:{min_value}")
                    # vertical_line = Span(location=min_value_idx, dimension='height', line_color='red', line_dash='dashed', line_width=2)
                    # vertical_line = Span(location=min_value_idx, dimension='height', line_color='red', line_dash='dashed', line_width=2)
                    # fig.add_layout(vertical_line)
                    fig.segment(
                        min_value_idx,
                        0,
                        min_value_idx,
                        min_value,
                        color="red",
                        line_width=3,
                        legend_label=f"Max Drawdown({min_value * 100:.1f}%)",
                    )

                    # Increase the range of the y-axis
                    extra_space = 0.052  # Adjust the amount of extra space you want
                    y_range = fig.y_range
                    # y_range.start = min_value - extra_space
                    y_range.end = max_value + extra_space

                    return fig
                except Exception as e:
                    tb_lineno = e.__traceback__.tb_lineno if e.__traceback__ else "unknown"
                    tb_filename = (
                        e.__traceback__.tb_frame.f_code.co_filename
                        if e.__traceback__ and e.__traceback__.tb_frame
                        else "unknown"
                    )
                    print(
                        f"Something went wrong when _plot_drawdown_section. Error:{str(e)} in line {tb_lineno} for file {tb_filename}."
                    )
                    return None

            def _plot_nav_section(height: float = 145):
                try:
                    equity = data["navs"].copy()
                    if "max_drawdown" in stats.index and stats.loc["max_drawdown"]:
                        max_dd = float(stats.loc["max_drawdown"])
                        dd_start, dd_end = stats.loc["max_drawdown_period"]
                        dd_timedelta_label = dd_end - dd_start

                        dd_start, dd_end = (
                            data[data["datetime_index"] == dd_start].index[0],
                            data[data["datetime_index"] == dd_end].index[0],
                        )
                        if np.isnan(dd_end):
                            dd_start = dd_end = equity.index[0]
                        else:
                            dd_start = equity[:dd_end].idxmax()
                            if dd_end != equity.index[-1]:
                                dd_end = np.interp(
                                    equity[dd_start],
                                    (equity[dd_end - 1], equity[dd_end]),
                                    (dd_end - 1, dd_end),
                                )
                                dd_end = int(dd_end)
                    if "max_runup" in stats.index and stats.loc["max_runup"]:
                        # max_ru = float(stats.loc["max_runup"])
                        ru_start, ru_end = stats.loc["max_runup_period"]
                        # ru_timedelta_label = ru_end - ru_start
                        ru_start, ru_end = (
                            data[data["datetime_index"] == ru_start].index[0],
                            data[data["datetime_index"] == ru_end].index[0],
                        )
                        if np.isnan(ru_end):
                            ru_start = ru_end = equity.index[0]
                        else:
                            ru_start = equity[:ru_end].idxmin()
                            if ru_end != equity.index[-1]:
                                ru_end = np.interp(
                                    equity[ru_start],
                                    (equity[ru_end - 1], equity[ru_end]),
                                    (ru_end - 1, ru_end),
                                )
                                ru_end = int(ru_end)
                    yaxis_label = "NAV"
                    source_key = "nav"
                    fig = new_indicator_figure(y_axis_label=yaxis_label, **(dict(height=height)))

                    if equity.empty:
                        return fig

                    source.add(equity, source_key)

                    tooltip_format = f"@{source_key}{{$ 0,0}}"
                    tick_format = "$ 0.0 a"
                    legend_format = "${:,.0f}"

                    r = fig.line("index", source_key, source=source, line_width=1.5, line_alpha=1)
                    tooltips = (
                        [
                            (yaxis_label, tooltip_format),
                            ("Benchmark", "@benchmark{+0,0.[000]%}"),
                        ]
                        if "benchmark" in data.columns
                        else [(yaxis_label, tooltip_format)]
                    )
                    set_tooltips(fig, tooltips, renderers=[r])
                    fig.yaxis.formatter = NumeralTickFormatter(format=tick_format)
                    # print('!!!',equity, equity.empty, equity.idxmax(),'!!!')
                    if equity.empty:
                        return fig
                    equity = equity.fillna(0.0)
                    argmax = equity.idxmax()
                    argmin = equity.idxmin()
                    # if not np.isnan(argmax):
                    fig.scatter(
                        argmax,
                        equity[argmax],
                        legend_label=f"Peak ({legend_format.format(equity[argmax])})",
                        color="cyan",
                        size=8,
                    )
                    fig.scatter(
                        argmin,
                        equity[argmin],
                        legend_label=f"Bottom ({legend_format.format(equity[argmin])})",
                        color="grey",
                        size=8,
                    )
                    fig.scatter(
                        index[-1],
                        equity.values[-1],
                        legend_label=f"Final ({legend_format.format(equity.iloc[-1])})",
                        color="blue",
                        size=8,
                    )
                    if "max_drawdown" in stats.index and stats.loc["max_drawdown"]:
                        fig.scatter(
                            dd_end,
                            equity.iloc[dd_end],
                            legend_label=f"Max Drawdown (-{100 * max_dd:.1f}%)",
                            color="red",
                            size=8,
                        )
                        lb = (
                            f"Max Dd Dur. ({dd_timedelta_label})".replace(", 0:00:00", "")
                            .replace(" 00:00:00", "")
                            .replace("(0 days ", "(")
                        )
                        fig.line(
                            [dd_start, dd_end],
                            [equity.iloc[dd_start], equity.iloc[dd_end]],
                            line_color="red",
                            line_width=2,
                            legend_label=lb,
                        )
                    # if "max_runup" in stats.index and stats.loc["max_runup"]:
                    #     fig.scatter(ru_end, equity.iloc[ru_end], legend_label="Max Runup (+{:.1f}%)".format(100 * max_ru), color="green", size=8)
                    #     lb = f"Max Ru Dur. ({ru_timedelta_label})".replace(", 0:00:00", "").replace(" 00:00:00", "").replace("(0 days ", "(")
                    #     fig.line([ru_start, ru_end], [equity.iloc[ru_start], equity.iloc[ru_end]], line_color="green", line_width=2, legend_label=lb)

                    fig.xaxis.formatter = _xaxis_formatter
                    fig.xaxis.visible = True

                    annualized_return = stats.loc["annualized_return"]
                    sharpe_ratio = stats.loc["sharpe_ratio"]
                    max_drawdown = stats.loc["max_drawdown"]

                    if annualized_return and sharpe_ratio and max_drawdown:
                        # fig.text(x=dates[-1], y=returns[-1], text=[f'APY: {annualized_return:.2%}'], text_font_size='12pt', text_baseline='middle', text_align='left', text_color='green')
                        # fig.text(x=dates[-1], y=returns[-1], text=[f'Sharpe Ratio: {sharpe_ratio:.2f}'], text_font_size='12pt', text_baseline='bottom', text_align='left', text_color='red')

                        label_text = f"APY: {annualized_return:.2%}\nSharpe Ratio: {sharpe_ratio:.2f}\nMax Dd: {max_drawdown:.2%}"
                        width = 1000
                        label = Label(
                            x=width - 430,
                            y=20,
                            x_units="screen",
                            y_units="screen",
                            text=label_text,
                            text_align="left",
                            background_fill_color="white",
                            background_fill_alpha=0.7,
                            text_font_size="10pt",
                        )
                        # label = Label(x=1, y=1, x_units='data', y_units='data',
                        #             text=label_text, text_color='black',
                        #             background_fill_color='white', background_fill_alpha=0.7,
                        #             text_font_size='12pt', text_align='right')

                        fig.add_layout(label)

                    return fig
                except Exception as e:
                    tb_lineno = e.__traceback__.tb_lineno if e.__traceback__ else "unknown"
                    tb_filename = (
                        e.__traceback__.tb_frame.f_code.co_filename
                        if e.__traceback__ and e.__traceback__.tb_frame
                        else "unknown"
                    )
                    print(
                        f"Something went wrong when _plot_nav_section. Error:{str(e)} in line {tb_lineno} for file {tb_filename}."
                    )
                    return None

            def _plot_turnover_section(xaxis_visible: bool = False, height: float = 145):
                try:
                    # holding_value = data["holding_value"]
                    # available_balance = data["available_balance"]
                    # investment = data["total_net_investment"]
                    turnover = data["total_turnover"]

                    fig = new_indicator_figure(y_axis_label="Ttl. Turnover", **(dict(height=height)))
                    if turnover.empty:
                        return fig

                    tick_format = "$ 0.0 a"
                    legend_format = "${:,.0f}"

                    r = fig.line(
                        "index",
                        "total_turnover",
                        source=source,
                        line_width=1.5,
                        line_alpha=1,
                        legend_label=f"Ttl. Turnover ({legend_format.format(turnover.iloc[-1])})",
                    )
                    # r_i = fig.line(
                    #     "index",
                    #     "total_net_investment",
                    #     source=source,
                    #     line_width=1.5,
                    #     line_alpha=1,
                    #     line_color="black",
                    #     legend_label=f"Net Investment ({legend_format.format(investment.iloc[-1])})",
                    # )
                    # r_h = fig.line("index", 'holding_value', source=source, line_width=1.5, line_alpha=1, line_color="cyan",legend_label=f"Holding Values ({legend_format.format(holding_value.iloc[-1])})")
                    # r_b = fig.line("index", 'available_balance', source=source, line_width=1.5, line_alpha=1, line_color="blue",legend_label=f"Balance({legend_format.format(available_balance.iloc[-1])})")

                    tooltips = [
                        ("Ttl. Turnover", "@total_turnover{$ 0,0}"),
                        ("Net Investment", "@total_net_investment{$ 0,0}"),
                        # ('Holding Values', f"@holding_value{{$ 0,0}}"),
                        # ('Balance', f"@available_balance{{$ 0,0}}")
                    ]
                    set_tooltips(fig, tooltips, renderers=[r])
                    fig.yaxis.formatter = NumeralTickFormatter(format=tick_format)
                    # print('!!!',turnover, turnover.empty, turnover.idxmax(),'!!!')
                    if turnover.empty:
                        return fig

                    if xaxis_visible:
                        fig.xaxis.formatter = _xaxis_formatter
                        fig.xaxis.visible = True
                    return fig
                except Exception as e:
                    tb_lineno = e.__traceback__.tb_lineno if e.__traceback__ else "unknown"
                    tb_filename = (
                        e.__traceback__.tb_frame.f_code.co_filename
                        if e.__traceback__ and e.__traceback__.tb_frame
                        else "unknown"
                    )
                    print(
                        f"Something went wrong when _plot_turnover_section. Error:{str(e)} in line {tb_lineno} for file {tb_filename}."
                    )
                    return None

            def _plot_pnl_section(height: float = 80):
                try:
                    fig = new_indicator_figure(y_axis_label="Profit / Loss", **(dict(height=height)))
                    fig.xaxis.formatter = _xaxis_formatter
                    fig.xaxis.visible = True
                    r = fig.vbar("index", BAR_WIDTH, "pnl_net", source=source, color=inc_cmap)
                    set_tooltips(fig, [("PNL", "@pnl_net{0.00}")], renderers=[r])
                    fig.yaxis.formatter = NumeralTickFormatter()
                    return fig
                except Exception as e:
                    tb_lineno = e.__traceback__.tb_lineno if e.__traceback__ else "unknown"
                    tb_filename = (
                        e.__traceback__.tb_frame.f_code.co_filename
                        if e.__traceback__ and e.__traceback__.tb_frame
                        else "unknown"
                    )
                    print(
                        f"Something went wrong when _plot_pnl_section. Error:{str(e)} in line {tb_lineno} for file {tb_filename}."
                    )
                    return None

            def _plot_monthly_return_section(m, y_axis_label, height: float = 150, in_percent: bool = True):
                try:
                    # print(m,'?!?!?!') # {'2023-11-30 23:30:00': 0.0, '2023-12-31 23:30:00': -0.0979, '2024-01-31 00:00:00': -0.03625} ?!?!?!

                    month_returns_df = pd.DataFrame.from_dict(m, orient="index", columns=["rate"])
                    month_returns_df.index = pd.to_datetime(month_returns_df.index)
                    month_returns_df["Year"] = month_returns_df.index.strftime("%Y")
                    month_returns_df["Month"] = month_returns_df.index.strftime("%b")
                    month_returns_df.rate = round(month_returns_df.rate * (100 if in_percent else 1), 2)
                    month_returns_df

                    years = list(month_returns_df.Year.unique())
                    months_map = {
                        "Jan": 1,
                        "Feb": 2,
                        "Mar": 3,
                        "Apr": 4,
                        "May": 5,
                        "Jun": 6,
                        "Jul": 7,
                        "Aug": 8,
                        "Sep": 9,
                        "Oct": 10,
                        "Nov": 11,
                        "Dec": 12,
                    }
                    months = list(month_returns_df.Month.unique())
                    months = sorted(months, key=lambda x: months_map[x])

                    df = month_returns_df
                    df["rate_str"] = df["rate"].astype(str) + ("%" if in_percent else "")
                    # print(df,'_plot_monthly_return_section')

                    # reshape to 1D array or rates with a month and year for each row.
                    # df = pd.DataFrame(data.stack(), columns=['rate']).reset_index()

                    # this is the colormap from the original NYTimes plot
                    # colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
                    # colors = ["#5df20c", "#8bf20c", "#c9d9d3", "#b1f20c", "#dff20c", "#f2e70c", "#cc7878", "#f2a60c", "#f2230c"]
                    colors = RdYlGn[11]
                    colors = list(reversed(colors))
                    mapper = LinearColorMapper(palette=colors, low=df.rate.min(), high=df.rate.max())

                    # fig = _figure(
                    #     # y_axis_label="Monthly Returns(%) ({0} - {1})".format(years[0], years[-1]),
                    #     y_axis_label="Monthly Returns(%)",
                    #     # width=800,
                    #     height=150,
                    #     # x_range=years, y_range=list(reversed(months)),
                    #     x_range=months,
                    #     y_range=years,
                    #     active_scroll="xwheel_zoom",
                    #     active_drag="xpan",
                    #     tooltips=[("Date", "@Month @Year"), ("Return", "@rate%")],
                    #     tools="xpan,xwheel_zoom,box_zoom,undo,redo,reset,save",
                    # )
                    # # fig.xaxis.visible = False
                    # fig.yaxis.minor_tick_line_color = None
                    fig = new_indicator_figure(
                        y_axis_label=y_axis_label,
                        x_range=months,
                        x_axis_type=None,
                        y_range=years,
                        **(dict(height=height)),
                    )  # tooltips=[("Date", "@Month @Year"), ("Return", "@rate%")],
                    fig.xaxis.visible = True

                    fig.grid.grid_line_color = None
                    fig.axis.axis_line_color = None
                    fig.axis.major_tick_line_color = None
                    # fig.axis.major_label_text_font_size = "7px"
                    fig.axis.major_label_standoff = 0
                    fig.xaxis.major_label_orientation = pi / 3

                    # fig.rect(x="Year", y="Month", width=1, height=1,
                    r = fig.rect(
                        x="Month",
                        y="Year",
                        width=1,
                        height=1,
                        source=df,
                        fill_color={"field": "rate", "transform": mapper},
                        line_color=None,
                    )
                    set_tooltips(
                        fig,
                        [
                            ("Date", "@Month @Year"),
                            ("Return", "@rate" + ("%" if in_percent else "")),
                        ],
                        renderers=[r],
                        include_dt=False,
                    )

                    labels = LabelSet(
                        x="Month",
                        y="Year",
                        text="rate_str",
                        level="glyph",
                        text_align="center",
                        text_font_size="9px",
                        y_offset=-7,
                        source=ColumnDataSource(df),
                    )  # render_mode="canvas")
                    fig.add_layout(labels)

                    color_bar = ColorBar(
                        color_mapper=mapper,
                        major_label_text_font_size="7px",
                        ticker=BasicTicker(desired_num_ticks=len(colors)),
                        formatter=PrintfTickFormatter(format="%f" + ("%%" if in_percent else "")),
                        label_standoff=6,
                        border_line_color=None,
                    )
                    fig.add_layout(color_bar, "right")

                    return fig
                except Exception as e:
                    tb_lineno = e.__traceback__.tb_lineno if e.__traceback__ else "unknown"
                    tb_filename = (
                        e.__traceback__.tb_frame.f_code.co_filename
                        if e.__traceback__ and e.__traceback__.tb_frame
                        else "unknown"
                    )
                    print(
                        f"Something went wrong when _plot_monthly_return_section. Error:{str(e)} in line {tb_lineno} for file {tb_filename}."
                    )
                    return None

            # def _plot_monthly_distribution_section(height: float = 450):
            #     try:
            #         ##  TODO: 1. if only 1mth, use dailu instead 2. step=0.01 flexiable
            #         # m = stats.monthly_changes
            #         m = stats.daily_changes
            #         month_returns_df = pd.DataFrame.from_dict(
            #             m, orient="index", columns=["rate"]
            #         )
            #         month_returns_df.index = pd.to_datetime(month_returns_df.index)
            #         month_returns_df["Year"] = month_returns_df.index.strftime("%Y")
            #         month_returns_df["Month"] = month_returns_df.index.strftime("%b")
            #         # month_returns_df.rate = round(month_returns_df.rate * 100, 2)

            #         _min = month_returns_df.rate.min()
            #         _max = month_returns_df.rate.max()

            #         edges = np.arange(_min, _max, 0.01)
            #         month_returns_df_group = month_returns_df.copy()
            #         month_returns_df_group["rank"] = pd.cut(
            #             month_returns_df_group.rate, bins=edges
            #         )  #
            #         month_returns_df_group = month_returns_df_group.groupby("rank").agg(
            #             {
            #                 "Year": "size",
            #                 "rate": "mean",
            #             }
            #         )
            #         hist_freq = month_returns_df_group["Year"].tolist()
            #         hist_mean = month_returns_df_group["rate"].tolist()
            #         print(
            #             f"_min:{_min}; _max:{_max}; month_returns_df:{month_returns_df} \nmonth_returns_df_group: {month_returns_df_group} \nhist_freq:{hist_freq}; hist_mean:{hist_mean}"
            #         )

            #         month_returns_df_group_neg = month_returns_df.copy()
            #         month_returns_df_group_neg = month_returns_df_group_neg[
            #             month_returns_df_group_neg["rate"] < 0
            #         ]
            #         month_returns_df_group_neg["rank"] = pd.cut(
            #             month_returns_df_group_neg["rate"], bins=edges
            #         )  #
            #         month_returns_df_group_neg = month_returns_df_group_neg.groupby(
            #             "rank"
            #         ).agg(
            #             {
            #                 "Year": "size",
            #                 "rate": "mean",
            #             }
            #         )
            #         hist_freq_neg = month_returns_df_group_neg["Year"].tolist()
            #         hist_mean_neg = month_returns_df_group_neg["rate"].tolist()

            #         p6 = _figure(
            #             height=height,
            #             width=1000,
            #             #    x_axis_type="datetime",
            #             #         toolbar_location=None,
            #             #             tools="xpan",
            #             #             x_axis_location="above",
            #             #        background_fill_color="#efefef",
            #             #             x_range=(dates[1500], dates[2500])
            #         )
            #         p6.quad(
            #             top=hist_freq,
            #             bottom=0,
            #             left=edges[:-1],
            #             right=edges[1:],
            #             fill_color="skyblue",
            #             line_color="white",
            #             legend_label="Monthly Return Distribution",
            #         )

            #         p6.quad(
            #             top=hist_freq_neg,
            #             bottom=0,
            #             left=edges[:-1],
            #             right=edges[1:],
            #             fill_color="red",
            #             # fill_color={'field':'petal_width','transform':color_mapper},
            #             line_color="white",
            #             fill_alpha=0.5,
            #             legend_label="Monthly Return Distribution(Loss)",
            #         )

            #         vline = False
            #         p6.add_tools(
            #             HoverTool(
            #                 point_policy="follow_mouse",
            #                 tooltips=[
            #                     ("Freq", "@top"),
            #                     ("Range", "@left{0.2f} - @right{0.2f}"),
            #                 ],
            #                 mode="vline" if vline else "mouse",
            #                 # formatters={'@Date': 'datetime', }
            #             )
            #         )
            #         linked_crosshair = CrosshairTool(dimensions="both")
            #         p6.add_tools(linked_crosshair)
            #         p6.yaxis.axis_label = "Frequency"
            #         p6.toolbar.logo = None
            #         show(p6)
            #         return
            #         years = list(month_returns_df.Year.unique())
            #         months_map = {
            #             "Jan": 1,
            #             "Feb": 2,
            #             "Mar": 3,
            #             "Apr": 4,
            #             "May": 5,
            #             "Jun": 6,
            #             "Jul": 7,
            #             "Aug": 8,
            #             "Sep": 9,
            #             "Oct": 10,
            #             "Nov": 11,
            #             "Dec": 12,
            #         }
            #         months = list(month_returns_df.Month.unique())
            #         months = sorted(months, key=lambda x: months_map[x])

            #         df = month_returns_df
            #         df["rate_str"] = df["rate"].astype(str) + "%"

            #         # reshape to 1D array or rates with a month and year for each row.
            #         # df = pd.DataFrame(data.stack(), columns=['rate']).reset_index()

            #         # this is the colormap from the original NYTimes plot
            #         # colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
            #         # colors = ["#5df20c", "#8bf20c", "#c9d9d3", "#b1f20c", "#dff20c", "#f2e70c", "#cc7878", "#f2a60c", "#f2230c"]
            #         colors = RdYlGn[11]
            #         colors = list(reversed(colors))
            #         mapper = LinearColorMapper(
            #             palette=colors, low=df.rate.min(), high=df.rate.max()
            #         )

            #         # TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

            #         # fig = new_indicator_figure(y_axis_label="Monthly Returns(%)")
            #         # new_bokeh_figure = partial(
            #         #     _figure,
            #         #     x_axis_type="linear",
            #         #     width=width,
            #         #     height=400,
            #         #     tools="xpan,xwheel_zoom,box_zoom,undo,redo,reset,save",
            #         #     active_drag="xpan",
            #         #     active_scroll="xwheel_zoom",
            #         # )
            #         fig = _figure(
            #             # y_axis_label="Monthly Returns(%) ({0} - {1})".format(years[0], years[-1]),
            #             y_axis_label="Monthly Returns(%)",
            #             # width=800,
            #             height=150,
            #             # x_range=years, y_range=list(reversed(months)),
            #             x_range=months,
            #             y_range=years,
            #             active_scroll="xwheel_zoom",
            #             active_drag="xpan",
            #             # x_axis_location="above",
            #             # x_axis_type='datetime',
            #             #  width=900, height=400,
            #             tooltips=[("Date", "@Month @Year"), ("Return", "@rate%")],
            #             tools="xpan,xwheel_zoom,ywheel_zoom,box_zoom,undo,redo,reset,save",
            #         )
            #         # fig.xaxis.visible = False
            #         fig.yaxis.minor_tick_line_color = None

            #         # p = figure(title="Monthly Returns(%) ({0} - {1})".format(years[0], years[-1]),
            #         #         x_range=years, y_range=list(reversed(months)),
            #         #         x_axis_location="above", width=900, height=400,
            #         #     #     tools=TOOLS, toolbar_location='below',
            #         #         tooltips=[('date', '@Month @Year'), ('rate', '@rate%')])

            #         fig.grid.grid_line_color = None
            #         fig.axis.axis_line_color = None
            #         fig.axis.major_tick_line_color = None
            #         # fig.axis.major_label_text_font_size = "7px"
            #         fig.axis.major_label_standoff = 0
            #         fig.xaxis.major_label_orientation = pi / 3

            #         # fig.rect(x="Year", y="Month", width=1, height=1,
            #         fig.rect(
            #             x="Month",
            #             y="Year",
            #             width=1,
            #             height=1,
            #             source=df,
            #             fill_color={"field": "rate", "transform": mapper},
            #             line_color=None,
            #         )

            #         labels = LabelSet(
            #             x="Month",
            #             y="Year",
            #             text="rate_str",
            #             level="glyph",
            #             text_align="center",
            #             text_font_size="9px",
            #             y_offset=-7,
            #             source=ColumnDataSource(df),
            #         )  # , render_mode="canvas"
            #         fig.add_layout(labels)

            #         color_bar = ColorBar(
            #             color_mapper=mapper,
            #             major_label_text_font_size="7px",
            #             ticker=BasicTicker(desired_num_ticks=len(colors)),
            #             formatter=PrintfTickFormatter(format="%d%%"),
            #             label_standoff=6,
            #             border_line_color=None,
            #         )
            #         fig.add_layout(color_bar, "right")

            #         return fig
            #     except Exception as e:
            #         print(
            #             f"Something went wrong when _plot_monthly_distribution_section. Error:{str(e)} in line {e.__traceback__.tb_lineno} for file {e.__traceback__.tb_frame.f_code.co_filename}."
            #         )
            #         return None

            # def _plot_portfolio_composition(height: float = 222):
            #     try:
            #         # fig = _figure(
            #         #     title="Current Portfolio Composition (Top10)",
            #         #     height=222,
            #         #     active_scroll="xwheel_zoom",
            #         #     active_drag="xpan",
            #         #     tools="xpan,xwheel_zoom,box_zoom,undo,redo,reset,save",
            #         #     tooltips="@symbol: @weight%",
            #         #     x_range=(-1, 1.0),
            #         # )
            #         fig = new_indicator_figure(
            #             y_axis_label="Current Portfolio Composition (Top10)",
            #             height=height,
            #             x_range=(-1, 1.0),
            #             x_axis_type=None,
            #         )  # tooltips=@symbol: @weight%,

            #         if not current_weight_df.empty:
            #             current_weight_df["angle"] = (
            #                 current_weight_df["weight"]
            #                 / current_weight_df["weight"].sum()
            #                 * 2
            #                 * pi
            #             )
            #             # print(f"current_weight_df: {current_weight_df}")
            #             symbol_len = len(current_weight_df)
            #             current_weight_df["color"] = (
            #                 Category20c[symbol_len]
            #                 if 2 < symbol_len < 20
            #                 else sample(Turbo256, symbol_len)
            #             )

            #             r = fig.wedge(
            #                 x=0.0,
            #                 y=1,
            #                 radius=0.35,
            #                 start_angle=cumsum("angle", include_zero=True),
            #                 end_angle=cumsum("angle"),
            #                 line_color="white",
            #                 fill_color="color",
            #                 legend_field="symbol",
            #                 source=current_weight_df,
            #             )
            #             set_tooltips(
            #                 fig,
            #                 tooltips=[("Symbol", "@symbol"), ("Weight", "@weight%")],
            #                 renderers=[r],
            #                 include_dt=False,
            #             )

            #             fig.axis.axis_label = None
            #             fig.axis.visible = False
            #             fig.grid.grid_line_color = None

            #         return fig
            #     except Exception as e:
            #         print(
            #             f"Something went wrong when _plot_portfolio_composition. Error:{str(e)} in line {e.__traceback__.tb_lineno} for file {e.__traceback__.tb_frame.f_code.co_filename}."
            #         )
            #         return None

            # def _plot_portfolio_composition_hist():
            #     try:
            #         weights = []  # [{'BTCUSDT': 0.01, 'datetime':'2022-10-31 08:00:00+08:00'}, ]
            #         for pnl_snapshot in stats["pnl"]:
            #             #
            #             tmp = {
            #                 "datetime": pnl_snapshot["datetime"],
            #             }
            #             for pos in pnl_snapshot["positions"]:
            #                 symbol = pos["symbol"]
            #                 # tmp[symbol] = pos['weight_wo_cash']
            #                 if symbol in tmp:
            #                     tmp[symbol] += pos[
            #                         "weight_wo_cash"
            #                     ]  # weight / weight_wo_cash
            #                 else:
            #                     tmp[symbol] = pos[
            #                         "weight_wo_cash"
            #                     ]  # weight / weight_wo_cash
            #             # print(pnl_snapshot["datetime"], '>>>>',tmp)
            #             weights.append(tmp)

            #         weights_df = pd.DataFrame(weights).fillna(
            #             0.0
            #         )  # .set_index('datetime')
            #         # print(f"top10_symbols:{top10_symbols}; weights_df: \n{weights_df}")
            #         df = pd.DataFrame()
            #         if not weights_df.empty:
            #             # weights_df = weights_df[top10_symbols + ["datetime"]]  # NOTE: filter top 10
            #             weights_df = weights_df[
            #                 [
            #                     symbol
            #                     for symbol in top10_symbols
            #                     if symbol in weights_df.columns
            #                 ]
            #                 + ["datetime"]
            #             ]  # NOTE: filter top 10
            #             # print(top10_symbols, weights_df,'!!!!!')
            #             for symbol in top10_symbols:
            #                 if symbol in weights_df.columns:
            #                     weights_df[symbol] = round(weights_df[symbol] * 100, 2)

            #             weights_df["datetime"] = pd.to_datetime(weights_df["datetime"])
            #             weights_df = weights_df.set_index("datetime", drop=False)
            #             # print(weights_df, weights_df.columns,'!!!')
            #             # print('--->',data)

            #             df = weights_df
            #             df = df.loc[df.index.isin(data["datetime"])].copy()
            #             df.index.name = None
            #             df["datetime"] = df.index
            #             df["datetime_str"] = df.datetime.apply(
            #                 lambda x: x.strftime("%a %d %b %Y %H:%M:%S %p %Z")
            #             )
            #             df = df.reset_index(
            #                 drop=True
            #             )  # FIXME: x-axis's data in consist of index instead of using dt directly, so when reset_index, all the dt will start from 0
            #             df.index = data.loc[
            #                 data.datetime.isin(df["datetime"])
            #             ].index  # NOTE: This index match code fixed the above issue
            #             # print(df, df.columns,'!!!!!')

            #         fig = new_indicator_figure(
            #             y_axis_label="Hist Portfolio Composition (Top10)",
            #             height=300,
            #             y_range=(0, 110),
            #         )
            #         fig.xaxis.formatter = _xaxis_formatter
            #         fig.xaxis.visible = True

            #         names = top10_symbols
            #         if names:
            #             symbol_len = len(names)
            #             colors = (
            #                 Category20c[symbol_len]
            #                 if 2 < symbol_len < 10
            #                 else sample(Turbo256, symbol_len)
            #             )
            #             r = fig.varea_stack(
            #                 stackers=names,
            #                 x="index",
            #                 color=colors,
            #                 legend_label=names,
            #                 source=df,
            #             )
            #             # set_tooltips(fig=fig, tooltips= [(symbol, f"@{symbol}%") for symbol in names], renderers=r)
            #             # set_tooltips(fig=fig, tooltips= [(symbol, "@{"+symbol+"}%") for symbol in symbols], renderers=[r])

            #             # fig.legend.orientation = "horizontal"
            #             # fig.legend.background_fill_color = "#fafafa"

            #             formatters = {"@datetime": "datetime"}
            #             tooltips = [("Datetime", "@datetime_str")]
            #             for symbol in top10_symbols:
            #                 tooltips.append((symbol, f"@{symbol}%"))
            #             fig.add_tools(
            #                 HoverTool(tooltips=tooltips, formatters=formatters)
            #             )

            #         return fig
            #     except Exception as e:
            #         print(
            #             f"Something went wrong when _plot_portfolio_composition_hist. Error:{str(e)} in line {e.__traceback__.tb_lineno} for file {e.__traceback__.tb_frame.f_code.co_filename}."
            #         )
            #         return None

            def _plot_volume_section(feed_source, is_candlestick=None):
                try:
                    fig = new_indicator_figure(y_axis_label="Volume")
                    fig.xaxis.formatter = _xaxis_formatter
                    fig.xaxis.visible = True
                    r = fig.vbar(
                        "index",
                        BAR_WIDTH,
                        "volume",
                        source=feed_source,
                        color=ohlc_inc_cmap if is_candlestick else "blue",
                    )
                    set_tooltips(fig, [("Volume", "@volume{0.00 a}")], renderers=[r])
                    fig.yaxis.formatter = NumeralTickFormatter(format="0 a")
                    return fig
                except Exception as e:
                    tb_lineno = e.__traceback__.tb_lineno if e.__traceback__ else "unknown"
                    tb_filename = (
                        e.__traceback__.tb_frame.f_code.co_filename
                        if e.__traceback__ and e.__traceback__.tb_frame
                        else "unknown"
                    )
                    print(
                        f"Something went wrong when _plot_volume_section. Error:{str(e)} in line {tb_lineno} for file {tb_filename}."
                    )
                    return None

            def _plot_ohlcv(
                aio_symbol: str, df: pd.DataFrame, graph_data: dict[str, Any] = {}, is_candlestick: bool | None = None
            ) -> tuple[Any, Any] | None:
                """
                df - ohlc data
                data - hist return

                TODO: the data is not udpate ===> due to the statistic construct_data func isnt invoke, dt index isnt update.
                """
                try:
                    # print(f"IN plot ohlcv => df:{df};\ndata:{data}; len of df:{len(df)};len of data:{len(data)}; \ndf.index:{df.index}; \ndata['datetime']:{data['datetime']}")
                    df = df.loc[
                        df.index.isin(data["datetime"])
                    ].copy()  ### filter the datafeed datetime that is in the return data
                    df.index.name = None
                    df["datetime"] = df.index
                    df["datetime_str"] = df.datetime.apply(lambda x: x.strftime("%a %d %b %Y %H:%M:%S %p %Z"))
                    df = df.reset_index(
                        drop=True
                    )  # FIXME: x-axis's data in consist of index instead of using dt directly, so when reset_index, all the dt will start from 0
                    df.index = data.loc[
                        data.datetime.isin(df["datetime"])
                    ].index  # NOTE: This index match code fixed the above issue
                    # print('~~~~~~~~~',df.tail(5),'~~~~~~~~~')
                    if is_candlestick is None:
                        is_candlestick = len(df) < _MAX_CANDLES

                    # print('======> ',aio_symbol, df,'!!!!!', data)
                    feed_source = ColumnDataSource(df)
                    if is_candlestick:
                        feed_source.add(
                            (df.close >= df.open).values.astype(np.uint8).astype(str),
                            "ohlc_inc",
                        )
                        ohlc_extreme_values = df[["high", "low"]].copy(deep=False)
                    NBSP = "\N{NBSP}" * 4
                    ohlc_tooltips = [
                        ("x, y", NBSP.join(("$index", "$y{0,0.0[0000]}"))),
                        (
                            "OHLC",
                            NBSP.join(
                                (
                                    "$@open{0,0.0[0000]}",
                                    "$@high{0,0.0[0000]}",
                                    "$@low{0,0.0[0000]}",
                                    "$@close{0,0.0[0000]}",
                                )
                            ),
                        ),
                        ("Volume", "@volume{0,0}"),
                    ]
                    if _xrange is None:
                        fig_ohlc = new_bokeh_figure(y_axis_label=aio_symbol.replace("|", " - "))
                    else:
                        fig_ohlc = new_bokeh_figure(x_range=_xrange, y_axis_label=aio_symbol.replace("|", " - "))
                    fig_ohlc.xaxis.formatter = _xaxis_formatter
                    if graph_data:
                        r, ohlc_tooltips = _plot_custom_supplement(graph_data, fig_ohlc, feed_source, ohlc_tooltips)
                    if plot_candlestick and is_candlestick:
                        fig_ohlc.segment(
                            "index",
                            "high",
                            "index",
                            "low",
                            source=feed_source,
                            color="black",
                        )
                        r = fig_ohlc.vbar(
                            "index",
                            BAR_WIDTH,
                            "open",
                            "close",
                            source=feed_source,
                            line_color="black",
                            fill_color=ohlc_inc_cmap,
                        )
                    else:
                        r = fig_ohlc.line(
                            "index",
                            "close",
                            source=feed_source,
                            line_color="black",
                            line_width=2.5,
                            line_alpha=1,
                            legend_label="Close",
                        )
                    set_tooltips(fig_ohlc, ohlc_tooltips, vline=True, renderers=[r])
                    trades = self.trade_records.get(aio_symbol, [])
                    trades_df = pd.DataFrame(trades)
                    # print(f"[_plot_ohlcv - trade part] trades_df ori:{trades_df} \n df:{df}")
                    if not trades_df.empty and not df.empty:
                        try:
                            # print(f"action: {trades_df['action']}")
                            trades_df["side"] = trades_df["side"].apply(lambda x: x.name)
                            # trades_df["oc"] = np.where(trades_df["is_open"], "OPEN", "CLOSE")
                            # print(trades_df["side"],'????',trades_df.iloc[-1]["side"] , type(trades_df.iloc[-1]["side"]), trades_df["side"].astype(str))
                            # trades_df["oc"] = np.where(trades_df["is_open"], np.where(trades_df["side"]=="BUY", "OPEN LONG", "OPEN SHORT"), np.where(trades_df["side"]=="BUY", "CLOSE SHORT", "CLOSE LONG"))

                            trades_df = trades_df[
                                [
                                    "side",
                                    "traded_at",
                                    "price",
                                    "quantity",
                                    "action",
                                    "pnl",
                                ]
                            ]

                            # trades_df['traded_at'] = trades_df['traded_at'].astype(str)
                            # trades_df['traded_at'] = trades_df['traded_at'].str.split('+').str[0].str.split('.').str[0]
                            # BUG: Error:Tz-aware datetime.datetime cannot be converted to datetime64 unless utc=True
                            # BUG: Error:index is not a valid DatetimeIndex or PeriodIndex >>> https://stackoverflow.com/questions/26089670/unable-to-apply-methods-on-timestamps-using-series-built-ins
                            trades_df["traded_at"] = pd.to_datetime(trades_df["traded_at"])

                            # print(f"[_plot_ohlcv - trade part] trades_df traded_at:{trades_df['traded_at']}({trades_df['traded_at'].dtype}) || df['datetime']:{df['datetime']} ({df['datetime'].dtype})")

                            # trades_df.index = trades_df["traded_at"].apply(lambda x: df[df["datetime"] == x].index[0])
                            # trades_df.index = trades_df["traded_at"].apply(lambda x: (df["datetime"] - x).abs().idxmin())
                            trades_df.index = trades_df["traded_at"].apply(
                                lambda x: (df["datetime"] - x).abs().idxmin()
                            )

                            # print(f"[_plot_ohlcv - trade part] trades_df after update index:{trades_df}")

                            # trades_df.index = trades_df["traded_at"].apply(lambda x: df.index[df.datetime.get_loc(x, method="nearest")])  # find the closest????  # cloest_tick = df.index[df.index.get_loc(dt, method="nearest")]
                            trades_df.index.name = "index"
                            # trades_df["datetime_str"] = trades_df["traded_at"].apply(lambda x: x.strftime("%a %d %b %Y %H:%M:%S %p %Z"))
                            trades_df["datetime_str"] = trades_df["traded_at"].apply(
                                lambda x: x.strftime("%a %d %b %Y %H:%M:%S %p %Z")
                            )
                            trades_df["marker"] = trades_df["side"].apply(
                                lambda x: "inverted_triangle" if x in ["SELL", "SHORTSELL"] else "triangle"
                            )
                            # print(f"[_plot_ohlcv - trade part] trades_df before become datasource:{trades_df}")

                            size = trades_df["quantity"].abs()
                            size = np.interp(size, (size.min(), size.max()), (8, 20))
                            trades_source = ColumnDataSource(trades_df)
                            trades_source.add(size, "marker_size")
                            side_cmap = CategoricalColorMapper(
                                factors=trades_df["side"].unique(),
                                palette=["gold", "magenta"],
                            )
                            fs = fig_ohlc.scatter(
                                x="index",
                                y="price",
                                alpha=0.7,
                                marker="marker",
                                color={"field": "side", "transform": side_cmap},
                                line_color="black",
                                size="marker_size",
                                legend_field="side",
                                source=trades_source,
                            )
                            fs_tooltips = [
                                ("Side", "@side"),
                                ("O/C", "@action"),
                                ("Price", "$@price{0,0.00[0000]}"),
                                ("Quantity", "@quantity{0,0.00[0000]}"),
                                ("PNL", "$@pnl{0,0.0}"),
                            ]
                            set_tooltips(fig_ohlc, fs_tooltips, vline=False, renderers=[fs])
                        except Exception as e:
                            tb_lineno = e.__traceback__.tb_lineno if e.__traceback__ else "unknown"
                            tb_filename = (
                                e.__traceback__.tb_frame.f_code.co_filename
                                if e.__traceback__ and e.__traceback__.tb_frame
                                else "unknown"
                            )
                            print(
                                f"Something went wrong in _plot_ohlcv. Error:{str(e)} in line {tb_lineno} for file {tb_filename}. Encounter error when match trade with ohlc data. aio_symbol: {aio_symbol};  trades_df: {trades_df.tail(30)};  df: {df}"
                            )

                    fig_volume = _plot_volume_section(feed_source, is_candlestick=is_candlestick)
                    if is_candlestick:
                        feed_source.add(ohlc_extreme_values.min(1), "ohlc_low")
                        feed_source.add(ohlc_extreme_values.max(1), "ohlc_high")
                        custom_js_args = dict(ohlc_range=fig_ohlc.y_range, source=feed_source)
                        custom_js_args.update(volume_range=fig_volume.y_range)
                        fig_ohlc.x_range.js_on_change(
                            "end",
                            CustomJS(args=custom_js_args, code=_AUTOSCALE_JS_CALLBACK),
                        )
                    return fig_ohlc, fig_volume
                except Exception as e:
                    tb_lineno = e.__traceback__.tb_lineno if e.__traceback__ else "unknown"
                    tb_filename = (
                        e.__traceback__.tb_frame.f_code.co_filename
                        if e.__traceback__ and e.__traceback__.tb_frame
                        else "unknown"
                    )
                    print(
                        f"Something went wrong when _plot_ohlcv. Error:{str(e)} in line {tb_lineno} for file {tb_filename}."
                    )
                    return None

            def _plot_stats_table():
                try:
                    formater = HTMLTemplateFormatter(template=STATS_TABLE_TEMPLATE)  # , nan_format="N/A")

                    # print('???????_plot_stats_table??????????',stats, type(stats)) # panda.Series
                    # , 'closed_trades'   'pnl_daily',

                    # stats = self.stats
                    # print(f"[_plot_stats_table] start:{stats.start}({type(stats.start)}); end:{stats.end}({type(stats.end)}); ")
                    stats_df = (
                        stats.to_frame()
                        .drop(
                            [
                                "drawdown_series",
                                "rolling_sharpe",
                                "positions",
                                "trades",
                                "pnl",
                                "frequently_traded",
                                "monthly_changes",
                                "monthly_avg_risk_score",
                                "additional_capitals",
                                "daily_changes",
                                "monthly_sharpe",
                                "asset_values",
                            ],
                            axis=0,
                        )
                        .reset_index()
                    )  # axis=0 means drop rows
                    # print('???????_plot_stats_table??????????',stats_df, type(stats_df)) # panda.Series
                    stats_ds = ColumnDataSource(stats_df)
                    columns = [
                        TableColumn(field="index", title="<b>Metrics</b>"),  # default_sort='ascending'),
                        TableColumn(
                            field="value", title="<b>Values</b>", formatter=formater
                        ),  # TableColumn(field="value", title="Values" , formatter=NumberFormatter(format="0,0.00%", nan_format="N/A")),
                    ]
                    # for col in stats_df.columns:
                    #     # if col != 'name':
                    #     #     columns.append(TableColumn(field=c, title=c , formatter=NumberFormatter(format="$0,0", text_align='right', nan_format="N/A")))
                    #     columns.append(TableColumn(field=col, title=col, formatter=formater ))# , formatter=NumberFormatter(format="$0,0", text_align='right')))
                    # columns.append(TableColumn(field="Currency", title="Currency"))
                    if bokeh.__version__ <= "2.4.3":
                        data_table = DataTable(
                            source=stats_ds,
                            columns=columns,
                            width=260,
                            height=2200,
                            index_position=None,
                            height_policy="fit",
                        )  # width=800, height=800)  #  width=1200,
                    else:
                        data_table = DataTable(
                            source=stats_ds,
                            columns=columns,
                            width=260,
                            height=2200,
                            index_position=None,
                        )  # width=800, height=800)  #  height_policy="fit", width=1200,
                    return data_table
                except Exception as e:
                    tb_lineno = e.__traceback__.tb_lineno if e.__traceback__ else "unknown"
                    tb_filename = (
                        e.__traceback__.tb_frame.f_code.co_filename
                        if e.__traceback__ and e.__traceback__.tb_frame
                        else "unknown"
                    )
                    print(
                        f"Something went wrong when _plot_stats_table. Error:{str(e)} in line {tb_lineno} for file {tb_filename}."
                    )
                    return None

            end = None
            # for aio_symbol, df in self.feeds.items():
            for feed in self.feeds:
                aio_symbol = feed.aio_symbol
                try:
                    df = feed.dataframe
                    if df is None or df.empty or "close" not in df.columns:
                        continue
                    is_candlestick = True
                    for c in [
                        "open",
                        "high",
                        "low",
                    ]:
                        if c not in df.columns:
                            is_candlestick = False

                    # print(f">>>[plotter] {df.tail(5)}")
                    cols = df.columns
                    # print(aio_symbol, '\n',df, )
                    ### TODO: resample
                    if self.resample_rule:
                        df = df[~df.index.duplicated(keep="last")]
                        ### TODO: resample agg: {'open':'first','close':'last', 'high':'max', 'low':'min', 'volume', 'sum'}
                        df = (
                            df.resample(
                                self.resample_rule if self.resample_rule else "D",
                                label="right",
                                closed="right",
                            )
                            .agg(resample_agg_gen(cols))
                            .ffill()
                            .fillna(0)
                            .round(5)
                        )
                        # print(f">>>[plotter - after resample] {df.tail(5)}")

                    # print(f"aio_symbol:{aio_symbol}; data:\n{df}")
                    if df is None or df.empty:
                        continue
                    if aio_symbol in self.custom_plot.keys():
                        result = _plot_ohlcv(
                            aio_symbol,
                            df=df,
                            graph_data=self.custom_plot[aio_symbol],
                            is_candlestick=is_candlestick,
                        )  # FIXME: custom_plot input??? # is_candlestick=len(df) < _MAX_CANDLES
                    else:
                        result = _plot_ohlcv(
                            aio_symbol, df, is_candlestick=is_candlestick
                        )  # is_candlestick=len(df) < _MAX_CANDLES

                    if result is not None:
                        fig_ohlcv, fig_volumn = result
                        fig_ohlcs.extend([fig_ohlcv, fig_volumn])
                        if extra_ohlc or only_ohlc:
                            fig_ohlcs_only.extend([fig_ohlcv, fig_volumn])
                    # end = df.index[-1]
                except Exception as e:
                    tb_lineno = e.__traceback__.tb_lineno if e.__traceback__ else "unknown"
                    tb_filename = (
                        e.__traceback__.tb_frame.f_code.co_filename
                        if e.__traceback__ and e.__traceback__.tb_frame
                        else "unknown"
                    )
                    print(
                        f"Something went wrong in _plot_ohlcv_loop. Error:{str(e)} in line {tb_lineno} for file {tb_filename}. aio_symbol:{aio_symbol}; df:\n{df} \nencounter some error when plotting ohlc chart data:\n{data.tail(30)}"
                    )

            if not only_ohlc:
                fig_stats_tb = _plot_stats_table()
                # show(fig_stats_tb)  # FOR DEBUG
                fig_ret = _plot_return_section(height=145)  # 380
                # show(fig_ret)  # FOR DEBUG
                figs_above_ohlc.append(fig_ret)
                fig_pnl = _plot_pnl_section(height=90)  # 250
                # show(fig_pnl)  # FOR DEBUG
                figs_above_ohlc.append(fig_pnl)
                fig_turnover = _plot_return_section(is_return=True, relative_equity=True, height=330)  # 450
                # show(fig_turnover)  # FOR DEBUG
                figs_above_ohlc.append(fig_turnover)
                if self.in_nav:
                    fig_ret = _plot_nav_section(height=330)  # 450
                    figs_above_ohlc.append(fig_ret)

                fig_dd = _plot_drawdown_section(height=200)
                figs_above_ohlc.append(fig_dd)

                # fig_ret = _plot_turnover_section(height=145)  # 380
                # # show(fig_ret)  # FOR DEBUG
                # figs_above_ohlc.append(fig_ret)
                fig_mth_returns = _plot_monthly_return_section(stats.monthly_changes, y_axis_label="Monthly Returns(%)")
                # show(fig_mth_returns)  # FOR DEBUG
                figs_above_ohlc.append(fig_mth_returns)

                fig_mth_sharpe_returns = _plot_monthly_return_section(
                    stats.monthly_sharpe,
                    y_axis_label="Monthly Sharpe",
                    in_percent=False,
                )
                # show(fig_mth_sharpe_returns)  # FOR DEBUG
                figs_above_ohlc.append(fig_mth_sharpe_returns)

                # fig_mth_dist_returns = _plot_monthly_distribution_section()

                aio_symbols = [feed.aio_symbol for feed in self.feeds]
                for figure_key, v in self.custom_plot.items():
                    figure_name, height = figure_key.split("|")
                    if figure_name not in aio_symbols:
                        fig_ret = _plot_custom(figure_name, v, height=int(height))
                        if fig_ret:
                            figs_below_ohlc.append(fig_ret)

                # fig_port_comp = _plot_portfolio_composition()
                # # show(fig_port_comp)  # FOR DEBUG
                # figs_below_ohlc.append(fig_port_comp)

                # fig_port_comp_hist = _plot_portfolio_composition_hist()
                # # show(fig_port_comp_hist)  # FOR DEBUG
                # if fig_port_comp_hist:
                #     figs_below_ohlc.append(fig_port_comp_hist)

            plots = figs_above_ohlc + fig_ohlcs + figs_below_ohlc
            # plots = []
            linked_crosshair = CrosshairTool(dimensions="both")
            for f in plots:
                if f.legend:
                    f.legend.visible = show_legend
                    f.legend.location = "top_left"
                    f.legend.border_line_width = 1
                    f.legend.border_line_color = "#333333"
                    f.legend.padding = 5
                    f.legend.spacing = 0
                    f.legend.margin = 0
                    f.legend.label_text_font_size = "8pt"
                    f.legend.click_policy = "hide"
                f.min_border_left = 0
                f.min_border_top = 3
                f.min_border_bottom = 6
                f.min_border_right = 10
                f.outline_line_color = "#666666"
                # linked_crosshair = CrosshairTool(dimensions="both")
                f.add_tools(linked_crosshair)
                wheelzoom_tool = next(wz for wz in f.tools if isinstance(wz, WheelZoomTool))
                wheelzoom_tool.maintain_focus = False
            kwargs = {}
            if width is None:
                # kwargs["sizing_mode"] = "stretch_width"
                kwargs["sizing_mode"] = "scale_width"

            ohlc_fig = None
            if extra_ohlc or only_ohlc:
                try:
                    # if end is not None:
                    #     start = end - np.timedelta64(30, 'D')
                    #     start_idx = (data["datetime"] - start).abs().idxmin()
                    #     end_idx = (data["datetime"] - end).abs().idxmin()
                    #     # print(f"end:{end}; start:{start}; start_idx:{start_idx}; end_idx:{end_idx}")
                    #     for f in fig_ohlcs_only:
                    #         f.x_range.start = start_idx
                    #         f.x_range.end = end_idx
                    ohlc_fig = gridplot(
                        fig_ohlcs_only,
                        sizing_mode="stretch_width",
                        ncols=1,
                        toolbar_location="above",
                        toolbar_options=dict(logo=None),
                        merge_tools=True,
                    )
                    if only_ohlc:
                        self.doc.add_root(ohlc_fig)
                        _bokeh_reset(path.replace(".html", "_ohlc.html"))
                        save(ohlc_fig)
                        curstate().reset()
                        curdoc().clear()
                        return ohlc_fig

                except Exception:
                    import traceback

                    traceback.print_exc()

            graph_fig = fig = gridplot(
                plots,
                ncols=1,
                toolbar_location="above",
                toolbar_options=dict(logo=None),
                merge_tools=True,
                **kwargs,
            )
            # graph_fig = fig = gridplot(plots, ncols=1, toolbar_location="above", merge_tools=True, **kwargs)

            if self.draw_figure is not None:
                graph_fig = fig = column(fig, self.draw_figure, sizing_mode="stretch_both")

            fig = row(fig, fig_stats_tb, spacing=10, sizing_mode="stretch_width")
            self.doc
            try:
                if export_as_png:
                    export_png(fig, filename=path)
                else:
                    if ohlc_fig:
                        # doc1 = curdoc()
                        self.doc.add_root(ohlc_fig)
                        _bokeh_reset(path.replace(".html", "_ohlc.html"))
                        # show(ohlc_fig, browser=None)
                        save(ohlc_fig)
                        curstate().reset()
                        curdoc().clear()

                    self.doc.add_root(fig)

                    if data.empty:
                        print("No data to generate report")
                        return fig

                    script_bokeh, div_bokeh = components(fig)
                    resources_bokeh = CDN.render()
                    # table_html = stats.to_frame().to_html(classes="ui selectable celled table", header=False)
                    html = template.render(resources=resources_bokeh, script=script_bokeh, div=div_bokeh)

                    # print(f">>>> path:{path} | open_browser:{open_browser} | IS_JUPYTER_NOTEBOOK:{IS_JUPYTER_NOTEBOOK}")
                    _bokeh_reset_html(html, path)
                    if open_browser:
                        # print('??', ("file:///" + path), )
                        webbrowser.open_new_tab("file:///" + path)
                    #     if path:
                    #         webbrowser.open_new_tab("file:///" + path)
                    #     else:
                    #         show(fig)

                    elif open_browser is None:
                        try:
                            # curstate().reset()
                            # reset_output()
                            # if IS_JUPYTER_NOTEBOOK:
                            #     output_notebook()
                            if not IS_JUPYTER_NOTEBOOK and path:
                                webbrowser.open_new_tab("file:///" + path)
                            else:
                                show(graph_fig)
                        except Exception:
                            # import traceback
                            # traceback.print_exc()
                            if IS_JUPYTER_NOTEBOOK:
                                output_notebook()
                            show(graph_fig)
            except Exception:
                import traceback

                traceback.print_exc()

            return fig
        except Exception as e:
            # import traceback
            # traceback.print_exc()
            tb_lineno = e.__traceback__.tb_lineno if e.__traceback__ else "unknown"
            tb_filename = (
                e.__traceback__.tb_frame.f_code.co_filename
                if e.__traceback__ and e.__traceback__.tb_frame
                else "unknown"
            )
            print(
                f"Something went wrong when plot_interactive. Error:{str(e)} in line {tb_lineno} for file {tb_filename}."
            )
            return None


def plot_heatmaps(
    heatmap: pd.Series,
    agg: Callable[..., Any] | str,
    ncols: int,
    filename: str = "",
    plot_width: int = 1200,
    open_browser: bool = True,
) -> Any:
    if not (isinstance(heatmap, pd.Series) and isinstance(heatmap.index, pd.MultiIndex)):
        raise ValueError("heatmap must be heatmap Series as returned by `Backtest.optimize(..., return_heatmap=True)`")

    _bokeh_reset(filename)

    param_combinations = combinations(heatmap.index.names, 2)
    dfs = [heatmap.groupby(list(dims)).agg(agg).to_frame(name="_Value") for dims in param_combinations]
    plots = []
    cmap = LinearColorMapper(
        palette="Viridis256",
        low=min(df.min().min() for df in dfs),
        high=max(df.max().max() for df in dfs),
        nan_color="white",
    )
    for df in dfs:
        name1, name2 = df.index.names
        level1 = df.index.levels[0].astype(str).tolist()
        level2 = df.index.levels[1].astype(str).tolist()
        df = df.reset_index()
        df[name1] = df[name1].astype("str")
        df[name2] = df[name2].astype("str")

        fig = _figure(
            x_range=level1,
            y_range=level2,
            x_axis_label=name1,
            y_axis_label=name2,
            width=plot_width // ncols,
            height=plot_width // ncols,
            tools="box_zoom,reset,save",
            tooltips=[
                (name1, "@" + name1),
                (name2, "@" + name2),
                ("Value", "@_Value{0.[000]}"),
            ],
        )
        fig.grid.grid_line_color = None
        fig.axis.axis_line_color = None
        fig.axis.major_tick_line_color = None
        fig.axis.major_label_standoff = 0

        fig.rect(
            x=name1,
            y=name2,
            width=1,
            height=1,
            source=df,
            line_color=None,
            fill_color=dict(field="_Value", transform=cmap),
        )
        plots.append(fig)

    fig = gridplot(
        plots,
        ncols=ncols,
        toolbar_options=dict(logo=None),
        toolbar_location="above",
        merge_tools=True,
    )

    show(fig, browser=None if open_browser else "none")
    return fig
